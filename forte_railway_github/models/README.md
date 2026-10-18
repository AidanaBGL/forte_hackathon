# Exported CPU models (committed for the all-in-one Railway deploy)

Put the files produced by `python -m worker.vision.export --format onnx` here, then
commit them so Railway's GitHub build copies them into the image:

```
models/
  best.onnx            # YOLO detector  -> FORTE_DETECTOR_WEIGHTS=/app/models/best.onnx
  action.onnx          # action net     -> FORTE_ACTION_CHECKPOINT=/app/models/action.onnx
  action.meta.json     # classes/frames/size/normalization (required next to action.onnx)
```

`.gitignore` is configured to keep `models/*.onnx`; the `.meta.json` sidecar is tracked
normally. OpenVINO IR (`best_openvino_model/`, `action.xml`+`action.bin`) also works here
if you prefer that format.
