import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

from model import MODELS, build_model


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, iou_thr):
    order = np.argsort(-pred_conf)
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(pred_boxes), bool)
    for i in order:
        if len(gt_boxes) == 0:
            break
        cand = np.where((gt_cls == pred_cls[i]) & ~gt_used)[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= iou_thr:
            tp[i] = True
            gt_used[j] = True
    return tp, gt_used


def hungarian_class_accuracy(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, 0
    ious = iou_matrix(pred_boxes, gt_boxes)
    r, c = linear_sum_assignment(-ious)
    keep = ious[r, c] >= iou_thr
    r, c = r[keep], c[keep]
    return int((pred_cls[r] == gt_cls[c]).sum()), len(r)


def draw(img_path, gt, preds, names, out_path):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), _ in gt:
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=1)
    for (x1, y1, x2, y2), cls, conf in preds:
        d.rectangle([x1, y1, x2, y2], outline=(230, 30, 30), width=2)
        d.text((x1, max(y1 - 10, 0)), f"{names[cls]} {conf:.2f}", fill=(255, 255, 0))
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", choices=MODELS, default="base")
    ap.add_argument("--resolution", type=int, default=None, help="resolution the checkpoint was trained at")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--conf", type=float, default=0.5, help="operating threshold for precision/recall/accuracy")
    ap.add_argument("--map-conf", type=float, default=0.01, help="low threshold for mAP")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n-vis", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only first N test images (debug)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "vis").mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.dataset) / "test"
    gt_json = test_dir / "_annotations.coco.json"

    coco_gt = COCO(str(gt_json))
    cat_ids = [c["id"] for c in coco_gt.dataset["categories"] if c["id"] != 0]
    names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    name_to_id = {v: k for k, v in names.items()}

    model = build_model(args.size, checkpoint=args.checkpoint, resolution=args.resolution)

    img_ids = sorted(coco_gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]

    coco_dets = []                      
    TP = FP = FN = 0
    acc_correct = acc_pairs = 0
    per_cls = defaultdict(lambda: [0, 0, 0])     
    per_image_fn = []
    vis_budget = args.n_vis

    for k, img_id in enumerate(img_ids):
        info = coco_gt.imgs[img_id]
        path = test_dir / info["file_name"]

        det = model.predict(str(path), threshold=args.map_conf)
        boxes = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
        conf = np.asarray(det.confidence, dtype=float)
        if det.data and "class_name" in det.data:
            cls = np.array([name_to_id.get(str(n), -1) for n in det.data["class_name"]], dtype=int)
        else:
            cls = np.asarray(det.class_id, dtype=int)
        ok = np.isin(cls, cat_ids)
        boxes, conf, cls = boxes[ok], conf[ok], cls[ok]

        for b, s, c in zip(boxes, conf, cls):
            coco_dets.append({"image_id": img_id, "category_id": int(c),
                              "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                              "score": float(s)})

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        gt_boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                              a["bbox"][1] + a["bbox"][3]] for a in anns], float).reshape(-1, 4)
        gt_cls = np.array([a["category_id"] for a in anns], int)

        sel = conf >= args.conf
        pb, pc, ps = boxes[sel], cls[sel], conf[sel]
        tp, gt_used = greedy_match(pb, pc, ps, gt_boxes, gt_cls, args.iou)
        TP += int(tp.sum()); FP += int((~tp).sum()); FN += int((~gt_used).sum())
        for c in cat_ids:
            per_cls[c][0] += int((tp & (pc == c)).sum())
            per_cls[c][1] += int((~tp & (pc == c)).sum())
            per_cls[c][2] += int((~gt_used & (gt_cls == c)).sum())
        per_image_fn.append((info["file_name"], int((~gt_used).sum()), int((~tp).sum())))

        n_ok, n_pairs = hungarian_class_accuracy(pb, pc, gt_boxes, gt_cls, args.iou)
        acc_correct += n_ok; acc_pairs += n_pairs

        if vis_budget > 0 and k % max(len(img_ids) // args.n_vis, 1) == 0:
            draw(path, [(b, c) for b, c in zip(gt_boxes, gt_cls)],
                 [(b, c, s) for b, c, s in zip(pb, pc, ps)], names, out / "vis" / info["file_name"])
            vis_budget -= 1

        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(img_ids)} images")

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    accuracy = acc_correct / acc_pairs if acc_pairs else 0.0

    mAP50 = mAP5095 = 0.0
    per_class_ap50 = {}
    if coco_dets:
        coco_dt = coco_gt.loadRes(coco_dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = img_ids
        ev.params.catIds = cat_ids
        ev.params.maxDets = [10, 100, 500]       
        ev.evaluate(); ev.accumulate(); ev.summarize()
        mAP5095, mAP50 = float(ev.stats[0]), float(ev.stats[1])
        prec = ev.eval["precision"]              
        for ki, c in enumerate(cat_ids):
            p = prec[0, :, ki, 0, -1]            
            p = p[p > -1]
            per_class_ap50[names[c]] = float(p.mean()) if p.size else 0.0

    per_class = {}
    for c in cat_ids:
        tp_, fp_, fn_ = per_cls[c]
        per_class[names[c]] = {
            "TP": tp_, "FP": fp_, "FN": fn_,
            "precision": tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0,
            "recall": tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0,
            "AP50": per_class_ap50.get(names[c], 0.0),
        }

    worst = sorted(per_image_fn, key=lambda t: -(t[1] + t[2]))[:10]
    result = {
        "settings": {"conf": args.conf, "iou": args.iou, "n_images": len(img_ids)},
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision, "recall": recall,
        "classification_accuracy_after_hungarian": accuracy,
        "mAP@0.5": mAP50, "mAP@0.5:0.95": mAP5095,
        "per_class": per_class,
        "worst_images_for_error_analysis": [{"file": f, "missed": m, "false_positives": p} for f, m, p in worst],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n=== Test results ===")
    print(f"precision {precision:.4f} | recall {recall:.4f} | accuracy {accuracy:.4f} | "
          f"mAP@0.5 {mAP50:.4f} | mAP@0.5:0.95 {mAP5095:.4f}")
    print(f"{'class':<16}{'P':>7}{'R':>7}{'AP50':>7}{'FN':>8}")
    for n, m in per_class.items():
        print(f"{n:<16}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['AP50']:>7.3f}{m['FN']:>8d}")
    print(f"\nSaved {out/'metrics.json'} and visualisations in {out/'vis'}")


if __name__ == "__main__":
    main()
