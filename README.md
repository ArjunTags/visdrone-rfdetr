# RF-DETR object detection on VisDrone2019-DET

Fine-tunes [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow's real-time detection transformer) on the
[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) drone-image dataset (10 classes) and evaluates it with
precision, recall, classification accuracy and mAP. Course final project (CO2.2).

## Repository layout

| Path | Purpose |
|---|---|
| `src/convert_visdrone.py` | Data loading and preprocessing: VisDrone text annotations to COCO JSON, cleaning, 70/15/15 split |
| `src/model.py` | Model definition: builds RF-DETR (Nano to Large) from pretrained or fine-tuned weights |
| `src/train.py` | Training: AdamW, EMA, gradient accumulation, early stopping, checkpointing |
| `src/evaluate.py` | Evaluation: IoU matching, precision, recall, Hungarian-matched accuracy, mAP, per-class table, prediction images |
| `notebooks/RFDETR_VisDrone_Final_Project.ipynb` | End-to-end notebook (two trials, curves, confusion matrix, error analysis) |
| `requirements.txt` | Python dependencies |

## Setup

Needs Python 3.10+ and an NVIDIA GPU (8 GB is enough with the settings below).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate

# Install a CUDA build of PyTorch FIRST (plain pip may give a CPU-only build on Windows).
# Pick the command for your CUDA version at https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

On Linux/WSL you also need a C compiler (used by Triton): `sudo apt install -y build-essential`

## Running

**1. Get the data.** Download the `train`, `val` and `test-dev` zips of VisDrone2019-DET from the
[official page](https://github.com/VisDrone/VisDrone-Dataset) and unzip them into one folder so you have
`VisDrone2019-DET-train/`, `VisDrone2019-DET-val/` and `VisDrone2019-DET-test-dev/`, each with `images/` and `annotations/`.
(`test-challenge` has no public labels and is not used.)

**2. Convert to COCO and split.**

```bash
python src/convert_visdrone.py --src /path/to/VisDrone --out data/visdrone_coco --split resplit
```

`resplit` pools all labelled images and cuts 70/15/15 with a fixed seed. `--split official` keeps VisDrone's own split.
Category 0 (ignored regions) and 11 (others) are dropped; boxes are clipped to the image.

**3. Train.**

```bash
python src/train.py --dataset data/visdrone_coco --out runs/trial1 --epochs 3 --batch-size 2 --grad-accum 8 --lr 1e-4 --resolution 560
python src/train.py --dataset data/visdrone_coco --out runs/trial2 --epochs 3 --batch-size 1 --grad-accum 16 --lr 1e-4 --resolution 728
```

Effective batch size = `batch-size x grad-accum` = 16 in both. The main change between trials is input resolution
(must be divisible by 56 for the Base model); batch size and accumulation were adjusted to fit an 8 GB GPU.
The reported results use 3 epochs per trial because of time limits, so the model is under-trained; raise `--epochs`
(for example to 20) for a longer run. Early stopping watches validation mAP; the best weights are saved as
`runs/<trial>/checkpoint_best_total.pth` and per-epoch metrics in `runs/<trial>/metrics.csv`.
Low on GPU memory? Lower `--batch-size`, raise `--grad-accum` to match, or add `--grad-checkpointing`.
Training hangs on Windows? Add `--num-workers 0`.

**4. Evaluate on the test split.**

```bash
python src/evaluate.py --dataset data/visdrone_coco --checkpoint runs/trial1/checkpoint_best_total.pth \
    --size base --resolution 560 --out eval_trial1
python src/evaluate.py --dataset data/visdrone_coco --checkpoint runs/trial2/checkpoint_best_total.pth \
    --size base --resolution 728 --out eval_trial2
```

`--resolution` must match the value used in training. Each run writes `<out>/metrics.json` (precision, recall,
accuracy, mAP, per-class numbers, worst images) and annotated test images in `<out>/vis/`
(green = ground truth, red = prediction).

**Notebook route.** `notebooks/RFDETR_VisDrone_Final_Project.ipynb` does all of the above with plots (training curves,
confusion matrix, error analysis). It runs locally from the `notebooks/` folder and keeps its data and runs in
`notebooks/work/`. Select the `.venv` kernel before running.

## Metric definitions

At IoU >= 0.5 and confidence >= 0.5: a detection is a true positive if it matches an unmatched ground-truth box of the
same class. Precision = TP/(TP+FP), recall = TP/(TP+FN). Accuracy pairs boxes one-to-one with the Hungarian algorithm
ignoring class and reports the share of pairs with the correct class. mAP uses `pycocotools` over all detections with
confidence >= 0.01 (up to 500 per image, since VisDrone frames are crowded).

## Results

Test set (1,295 images), 3 epochs per trial, confidence threshold 0.5 for precision, recall and accuracy:

| Trial | Resolution | mAP@50 | mAP@50:95 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| 1 | 560 | 0.386 | 0.218 | 0.870 | 0.354 | 0.918 |
| 2 | 728 | 0.448 | 0.260 | 0.874 | 0.414 | 0.923 |

Trial 2 is the reported model. See the project report for the full analysis.

## Notes

- RF-DETR itself (backbone, deformable-attention transformer, Hungarian matcher, losses) comes from the `rfdetr` package;
  this repo configures and fine-tunes it rather than re-implementing it.
- Written against `rfdetr` 1.11.x. Its training API has changed between releases, so pin the version if you need to reproduce exactly.

## References

Robinson et al. (2025) RF-DETR: Neural architecture search for real-time detection transformers, arXiv:2511.09554.
Carion et al. (2020) DETR, ECCV. Zhu et al. (2021) Deformable DETR, ICLR. Du et al. (2019) VisDrone-DET2019, ICCVW.
