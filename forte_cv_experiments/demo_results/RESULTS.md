# CPU sanity experiment on uploaded 42 videos

Dataset: 42 files, 36 unique videos for train/CV, 4 held-out test (X-15, X-16, X-22, X-23). 2 exact duplicate files excluded from training as duplicates. Run only with hand-crafted motion and appearance statistics; no pretrained detectors or video encoders were used. Windows: 4 seconds; stride: 5 seconds; 5 sampled frames/window, max frame side 640. Full 42-video feature extraction on CPU took ~72.1 seconds. This is a **smoke/baseline**, not final quality or fair comparison with sophisticated models.

| Model | CV AUC | CV accuracy | Test accuracy | Test violation recall | Test false positive rate |
|---|---:|---:|---:|---:|---:|
| E0 motion video classifier | 0.579 | 0.639 | 0.50 | 1.00 (2/2) | 1.00 (2/2 false alerts) |
| E1 weak temporal classifier | 0.344 | 0.528 | 0.50 | 1.00 (2/2) | 1.00 (2/2 false alerts) |

E1 flagged 17s in X-15 (ground truth 15s) and 17s in X-22 (ground truth 14s). The classifier also falsely flagged X-16 and X-23. Results are **not production acceptable** and especially unstable because test size is four. Data shared with the assistant has now been evaluated on the holdout, so use a further unseen set for truly blind model selection.

*Do not use the demo predictions as training action labels.*
