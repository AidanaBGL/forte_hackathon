# ForteBank — YOLO26 Object Detection Benchmark

Separate from the action-recognition project. Trains detection models only on labeled **images** and their YOLO bounding-box labels, not directly on MP4 files.

## Supported dataset layout (Roboflow YOLO26 export)

```text
dataset/
├── data.yaml
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
├── valid/
│   ├── images/*.jpg
│   └── labels/*.txt
└── test/
    ├── images/*.jpg
    └── labels/*.txt
```

Other common YOLO layouts (`images/train`, `labels/train`) also work if `data.yaml` points to them. A detection label row is `class_id cx cy w h` in **normalized** coordinates. You cannot train detection directly from video action labels such as `cash_counting_machine` with only start/end frames; this project requires bounding boxes for objects.

### Installation (Tesla V100, driver CUDA capability 12.8)

Use an isolated Python 3.12 environment. **Do not upgrade the server driver or alter existing CUDA deployments**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

If the `ultralytics` version installed in the venv cannot load `yolo26n.pt`, update `ultralytics` **inside the venv**. YOLO26 is officially supported by current Ultralytics releases. For some CUDA wheel configurations the PyTorch wheel may require a compatible NVIDIA driver; check your environment rather than treating the tested import as enough.

## 1. Audit labels/splits before training

```bash
python run.py inspect --data /hdd/home/ABaglanova/forte_hack/det_dataset --output runs/audit
```

The audit reports class frequency, missing test split, invalid label coordinates, exact-image SHA collisions across splits, and a heuristic warning for adjacent frames from the same source video crossing splits. If there is an exact-image collision, **resolve it** instead of hiding the warning. For video-derived datasets, group all frames from one source video together when splitting train/val/test. The name-based group heuristic is not exhaustive — verify the Roboflow export manually.

The script writes an absolute-path `data_resolved.yaml`, so Roboflow's confusing relative `../train/images` references are normalized for use on your server.

## 2. Run 3 quick experiments

```bash
CUDA_VISIBLE_DEVICES=0 python run.py sweep \
  --data /hdd/home/ABaglanova/forte_hack/det_dataset \
  --config experiments.json --output runs/detection \
  --limit 3 --epochs 40 --batch 4 --device 0
```

**Training is sequential**; avoids simultaneously occupying both busy V100s. Adjust batch if CUDA out of memory, e.g. `--batch 2`. If the second GPU is more free later, change `CUDA_VISIBLE_DEVICES=1` with `--device 0`.

Inspect `runs/detection/leaderboard.csv`. Rankings are by validation **mAP50-95**. Per-class validation AP CSV and official Ultralytics confusion plots are saved in experiment folders.

## 3. Full benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python run.py sweep \
 --data /hdd/home/ABaglanova/forte_hack/det_dataset \
 --output runs/detection_full --epochs 70 --batch 4 --device 0
```

Eight experiments: YOLO26n/s, no/light/mosaic augmentation, image resolution 640/960, different `cls`/`box` loss weights, AdamW vs SGD. YOLO26's `dfl` parameter is not a DFL classification loss: its newer regression machinery is architecture-specific. These experiments **do not** implement an independently substituted Focal Loss; they vary supported training loss gains and optimization settings. No benchmark outcome is claimed before training on your dataset.

Train one experiment:

```bash
python run.py train --data /path/to/dataset --name e05_y26n_960 --output runs/one --epochs 70 --batch 2
```

## 4. Evaluate winner ONCE on held-out test

```bash
python run.py test --data /path/to/dataset \
 --checkpoint runs/detection_full/e01_y26n_light/weights/best.pt \
 --output runs/final_test --imgsz 640 --batch 4 --device 0
```

Select the checkpoint according to validation performance **before** running test. Set `--imgsz` to the actual training resolution (e05 uses 960). Outputs test `mAP50`, `mAP50-95`, precision, recall, per-class AP, confusion matrix plots.

## 5. Prediction + visual boxes

```bash
python run.py predict \
 --checkpoint runs/detection_full/e01_y26n_light/weights/best.pt \
 --source /path/to/test/images --output runs/predictions --device 0 --imgsz 640
```

Saves annotated images and `.txt` predictions.

## Notes for the bank use case

* Do not substitute video-level action segments for detection boxes. They are separate labels.
* The frame-level detection task: `cash`, `document`, `cash_bag`, `counting_machine`, `safe`, `laptop`, `door`, `phone` plus optionally `person`.
* The state task `safe_open`/`safe_closed` and `laptop_open`/`laptop_closed` can be a separate crop classifier or attributes; the dataset must actually contain such state labels to train them.
* Model performance on near-duplicate frames from the same video is misleading unless all video frames are grouped into the same split. Four withheld images aren't a robust test; reserve at least several full source videos if possible.
* Keep video and image annotations local if hackathon terms prohibit sharing with external services.
* For closed-source banking deployment, review the license of Ultralytics and each checkpoint (AGPL-3.0 / Enterprise licensing concerns).

Official reference: https://docs.ultralytics.com/models/yolo26/
