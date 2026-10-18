# ForteBank — video action recognition benchmark

**Dataset:** Roboflow Video COCO (`videos`, `segments`, `categories`), `.mp4` or `.mov`.

The example segment `cash_counting_machine` from `start_frame=0` to `end_frame=698` in a ~23s video **marks the entire video with one action label**. It does NOT mean the model knows on which second the counting actually occurs. This project first trains segment/video classification, not precise action detection. If you need event timecodes for Datathon, annotate actual start/end times or use this model only as a proposal generator.

## 1. Place data

Either Roboflow-provided partitions:

```
data/
  train/_annotations.coco.json
  train/-43.mp4
  valid/_annotations.coco.json
  valid/-44.mp4
  test/_annotations.coco.json
  test/-45.mp4
```

Or one unsplit export:

```
data/
  _annotations.coco.json
  -43.mp4
  -44.mp4
  ...
```

For the second layout, the script splits **by distinct source video content**, not frames or clips; SHA-256 identical videos remain together. Default **4 test videos**, 20% of the remaining videos for validation, rest training. With very rare actions, 4 test samples may not cover all classes. Inspect class support! For experiments, use train+validation only to choose the winner; test only once at the end.

### Important: SHA-256 duplicates across Roboflow splits

If `prepare` fails with `IDENTICAL videos leak across Roboflow splits`, it means your
train/valid/test directories contain **identical MP4 bytes**. This guard is deliberate.
You can keep the original splits and remove/move the duplicates manually, or instruct
this script to **replace all Roboflow splits** with new SHA-256-grouped splits:

```bash
python train.py prepare --data /path/to/dataset --manifest runs/data \
  --test-count 4 --val-fraction 0.2 --seed 42 --resplit-by-hash
```

`--resplit-by-hash` assigns all byte-identical copies to exactly one split,
removes repeated identical annotations, and keeps 4 **unique** test videos where feasible.
It **does not preserve** Roboflow's original split assignments. Inspect the
resulting `runs/data/manifest.csv` and `runs/data/dataset_info.json`. The selected
validation and test cases must not be used to tune models by inspection.
If only a single unique video supports a class, the class may not have enough data
for reliable train/val/test evaluation and you may need additional independent videos.

## 2. Setup

Use the server environment with working PyTorch and CUDA. Verify:

```bash
python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
pip install -r requirements.txt
```

Do not blindly reinstall torch in an environment that already has the correct CUDA build. Models download **pretrained weights once** unless you supply `--no-pretrained` (not recommended for only ~42 videos). All inference after weights are cached is local. Keep bank video data locally; do not upload it to public third-party services.

## 3. Prepare, validate labels, split

```bash
python train.py prepare --data /path/to/your/data --manifest runs/data --test-count 4 --seed 42
```

Optionally force exact four named holdout videos for an unsplit dataset:

```bash
python train.py prepare --data /path/to/data --manifest runs/data \
  --test-videos=-15.mp4,-16.mp4,-22.mp4,-23.mp4
```

If filenames use unusual prefixes, pass the **exact** names inside your JSON/video folder. For spaces or non-ASCII names, quote them; alternatively directly move those videos/JSON into a `test/` Roboflow export. For a dataset that already contains `train/valid/test`, the provided partitions are kept and the `--test-videos` argument is not used.

Check `runs/data/dataset_info.json` + `runs/data/manifest.csv`. If most segments cover >80% of the video, labels are **weak supervision**. Do not train with short random action crops as if every crop contained the same action.

## 4. Experiments: models, losses, augmentations

Model choices:

- `mobilenet_gru`: fast pretrained MobileNetV3-small encoder + bidirectional GRU;
- `resnet18_gru`: pretrained ResNet18 + bi-GRU;
- `r3d18`: video ResNet 3D on 16 ordered frames;
- `mc3_18`: mixed-convolution 3D ResNet;
- `r2plus1d18`: 2+1D convolution architecture.

Four losses: `ce`, `weighted_ce` (inverse-sqrt class support), `smooth_ce` (label smoothing) and `focal` (gamma=2). Augmentations: `none/light/strong`; consistent spatial/color transforms are applied to every frame in a clip. No temporal reversal and no default horizontal flip. `train_views=2` gives two stochastic samples of **the same annotated temporal segment**, not frame-level leakage into validation.

A single baseline:

```bash
python train.py train --manifest runs/data --model mobilenet_gru \
  --loss ce --aug light --epochs 20 --batch 4 \
  --frames 16 --size 128 --out runs/baseline
```

Run first three cheaper configurations:

```bash
python sweep.py --config experiments.json --manifest runs/data \
  --output runs/sweep --limit 3
```

Then complete the grid (11 experiments: 5 backbones, CE/weighted/focal/smoothing, none/light/strong augmentation, frozen vs fine-tune):

```bash
python sweep.py --config experiments.json --manifest runs/data \
  --output runs/full_grid
```

Grid ranks models by **VAL macro-F1**, then accuracy. `runs/full_grid/leaderboard.csv` lists the experiments. The best checkpoint is `runs/full_grid/<winner>/best.pt`.

Evaluate held-out test **only when you have finished model selection**:

```bash
python train.py test --manifest runs/data \
  --checkpoint runs/full_grid/e01_mobilenet_ce_light/best.pt \
  --out runs/final_test
```

Or `python sweep.py ... --test-best`, but DO NOT rerun/change hyperparameters after seeing the test score. With 4 test videos, accuracy moves by 25 percentage points per video; accuracy alone is not a stable estimate.

Outputs: `val_predictions.csv`, `val_confusion.csv`, `val_per_class.csv`, `val_metrics.json`, `history.csv`, `best.pt`. After a test run: `test_predictions.csv`, `test_confusion.csv`, `test_per_class.csv`, `test_metrics.json`.

**GPU memory tips**: if OOM lower `--batch 1`, then `--frames 8`, then `--size 96`. With H200 use bigger batches if desired. `--freeze-backbone` is default for frame CNNs and explicitly enabled for initial 3D experiments. With more data, fine-tune 3D at LR ~2e-5.

## 5. Optional approximate sliding-window inference

```bash
python infer.py --checkpoint runs/baseline/best.pt --video /path/-43.mp4 \
  --window-sec 4 --stride-sec 2 --manifest runs/data \
  --out runs/inference_-43.json
```

If labels are whole-video, the script **refuses to generate falsely precise events** unless `--allow-weak-intervals` is explicitly given; such results are exploratory only. Fix start/end labels before judging event-localization quality.

## 6. Important caveats

- `images: []` and `annotations: []` in Roboflow **Video COCO is normal** for temporal action exports. The trainable labels are in `segments`.
- Category 0 `object` is generic placeholder, excluded if not an action segment.
- An annotated segment can have different length, several segments can be in one video; split at VIDEO level, not segment level.
- A full-video label cannot verify procedure completeness/order; for those tasks, later train a temporal model on **actual actions with start/end annotations**, then use a state machine with applicable rules.
- Bank safety rules may require nonvisual data. Visible typing != verified computer registration; closing laptop != confirmed session lock.
- If an action has only one independent training video, model accuracy for that class is not trustworthy. Record data coverage and consider filming additional allowed examples.

## 7. Local health check (no weights/no bank video required)

```bash
python sanity_check.py
```

Creates temporary synthetic MP4 Video COCO data, verifies JSON parsing, data loading, all five model forward passes, and all four losses. Full mobile training was also smoke-tested on synthetic video; this does not establish bank-dataset accuracy.
