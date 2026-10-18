"""Export the trained detector and action model to CPU-optimized formats.

Run this ONCE on a machine that has the trained weights plus torch/ultralytics.
It produces ONNX and/or OpenVINO artifacts that the worker can serve on a plain CPU
(e.g. a Railway container) with no GPU and no PyTorch at inference time for the action net.

Examples
--------
    # Both models to OpenVINO IR (recommended for Intel CPUs):
    python -m worker.vision.export \
        --detector runs/detection/e02_y26n_mosaic/weights/best.pt \
        --action   runs/experiments/e02_mobilenet_ce_strong/best.pt \
        --out models --format openvino --imgsz 640

    # ONNX only (portable, runs under onnxruntime anywhere):
    python -m worker.vision.export --action best.pt --out models --format onnx

Then point the worker at the exported files:
    FORTE_DETECTOR_WEIGHTS=models/best_openvino_model
    FORTE_ACTION_CHECKPOINT=models/action.xml      # or action.onnx
    FORTE_DEVICE=cpu
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def export_detector(weights, out_dir, fmt, imgsz, int8):
    """Ultralytics handles YOLO export natively; the worker loads the result via YOLO()."""
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    produced = []
    for one in (['onnx', 'openvino'] if fmt == 'both' else [fmt]):
        # half=False keeps FP32 — correct for CPU. int8 (OpenVINO) needs calibration data.
        path = model.export(format=one, imgsz=imgsz, dynamic=False, half=False,
                            int8=int8 if one == 'openvino' else False)
        src = Path(path)
        dst = out_dir / src.name
        if src.resolve() != dst.resolve():
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(src), str(dst))
        produced.append(dst)
        print(f'[detector] {one} -> {dst}')
    return produced


def export_action(checkpoint, out_dir, fmt, name, opset):
    """Trace the torch action net to ONNX; optionally convert ONNX -> OpenVINO IR.

    A <name>.meta.json sidecar carries classes/frames/size/normalization so the runtime
    backends never need the original checkpoint or the model-building code.
    """
    import torch
    from .action_arch import make_model
    out_dir.mkdir(parents=True, exist_ok=True)

    cp = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    classes = list(cp['classes'])
    frames, size = int(cp['frames']), int(cp['size'])
    model, _ = make_model(cp['model'], len(classes), pretrained=False,
                          freeze_backbone=cp.get('freeze_backbone'))
    model.load_state_dict(cp['state_dict'], strict=True)
    model.eval()

    meta = {'classes': classes, 'frames': frames, 'size': size,
            'normalization': cp['normalization'], 'model': cp['model']}
    meta_path = out_dir / f'{name}.meta.json'
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[action] metadata -> {meta_path}')

    # Fixed [1, C, T, H, W]; the worker always infers one window at a time.
    dummy = torch.randn(1, 3, frames, size, size)
    onnx_path = out_dir / f'{name}.onnx'
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(onnx_path),
                          input_names=['clip'], output_names=['logits'],
                          opset_version=opset, do_constant_folding=True)
    print(f'[action] onnx -> {onnx_path}')
    produced = [onnx_path]

    if fmt in ('openvino', 'both'):
        import openvino as ov
        ov_model = ov.convert_model(str(onnx_path))
        xml_path = out_dir / f'{name}.xml'
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
        # Keep the sidecar discoverable next to the IR file too.
        shutil.copyfile(meta_path, out_dir / f'{name}.meta.json')
        print(f'[action] openvino -> {xml_path}')
        produced.append(xml_path)
        if fmt == 'openvino':
            onnx_path.unlink(missing_ok=True)
            # torch may emit weights to an external-data sidecar; remove it too so no orphan is left.
            onnx_path.with_name(onnx_path.name + '.data').unlink(missing_ok=True)
            produced = [xml_path]
    return produced


def main():
    p = argparse.ArgumentParser(description='Export Forte models to ONNX / OpenVINO for CPU inference')
    p.add_argument('--detector', help='Path to the trained YOLO detector best.pt')
    p.add_argument('--action', help='Path to the trained action checkpoint best.pt')
    p.add_argument('--out', default='models', help='Output directory (default: models)')
    p.add_argument('--format', choices=['onnx', 'openvino', 'both'], default='openvino')
    p.add_argument('--imgsz', type=int, default=640, help='Detector inference size; set FORTE_DETECTOR_IMGSZ to match')
    p.add_argument('--action-name', default='action', help='Base name for exported action files')
    p.add_argument('--opset', type=int, default=17, help='ONNX opset for the action model')
    p.add_argument('--int8', action='store_true', help='INT8-quantize the OpenVINO detector (needs calibration data)')
    args = p.parse_args()

    if not args.detector and not args.action:
        p.error('Provide --detector and/or --action')
    out_dir = Path(args.out).resolve()
    if args.detector:
        export_detector(Path(args.detector), out_dir, args.format, args.imgsz, args.int8)
    if args.action:
        export_action(Path(args.action), out_dir, args.format, args.action_name, args.opset)
    print(f'\nDone. Artifacts in {out_dir}')
    print('Set FORTE_DEVICE=cpu, FORTE_DETECTOR_IMGSZ to the --imgsz used above, '
          'and point FORTE_DETECTOR_WEIGHTS / FORTE_ACTION_CHECKPOINT at the exported files.')


if __name__ == '__main__':
    main()
