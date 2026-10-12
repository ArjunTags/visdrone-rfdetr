import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

SRC_DIRS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def parse_annotation(txt_path: Path, w: int, h: int):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        x, y, bw, bh = (float(p) for p in parts[:4])
        cat = int(parts[5])
        if cat not in CLASSES:          
            continue
        x2, y2 = min(x + bw, w), min(y + bh, h)
        x, y = max(x, 0.0), max(y, 0.0)
        bw, bh = x2 - x, y2 - y
        if bw < 1 or bh < 1:
            continue
        boxes.append((x, y, bw, bh, cat))
    return boxes


def collect(src: Path):
    items = {}
    for key, folder in SRC_DIRS.items():
        img_dir = src / folder / "images"
        ann_dir = src / folder / "annotations"
        pairs = []
        for img in sorted(img_dir.glob("*.jpg")):
            pairs.append((img, ann_dir / (img.stem + ".txt")))
        items[key] = pairs
    return items


def write_split(name: str, pairs, out: Path, link: bool):
    split_dir = out / name
    split_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    for img_id, (img_path, ann_path) in enumerate(pairs, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        dst = split_dir / img_path.name
        if not dst.exists():
            if link:
                dst.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, dst)
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        for x, y, bw, bh, cat in parse_annotation(ann_path, w, h):
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": 0, "name": "objects", "supercategory": "none"}]
    categories += [{"id": i, "name": n, "supercategory": "objects"} for i, n in CLASSES.items()]

    coco = {"images": images, "annotations": annotations, "categories": categories}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"{name:>5}: {len(images):5d} images, {len(annotations):7d} boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", choices=["official", "resplit"], default="resplit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true", help="symlink images instead of copying (saves disk)")
    args = ap.parse_args()

    items = collect(args.src)
    if args.split == "official":
        plan = {"train": items["train"], "valid": items["val"], "test": items["test"]}
    else:
        pool = items["train"] + items["val"] + items["test"]
        random.Random(args.seed).shuffle(pool)
        n = len(pool)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        plan = {
            "train": pool[:n_tr],
            "valid": pool[n_tr:n_tr + n_va],
            "test": pool[n_tr + n_va:],
        }

    for name, pairs in plan.items():
        write_split(name, pairs, args.out, args.symlink)


if __name__ == "__main__":
    main()
