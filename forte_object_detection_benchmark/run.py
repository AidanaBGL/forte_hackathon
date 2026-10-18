#!/usr/bin/env python3
"""ForteBank YOLO26 object detection benchmark.

Commands: inspect, sweep, train, test, predict.
Never trains on val/test, never ranks models using the test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
DEFAULT_EXPERIMENTS = 'experiments.json'


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def write_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def image_list(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_split(root, yaml_dir, yml, split):
    rel = yml.get(split)
    if rel in (None, ''):
        return None
    if isinstance(rel, list) or str(rel).endswith('.txt'):
        raise ValueError(f'{split} contains a file list or multiple folders. This benchmark expects Roboflow image directories.')
    rel = Path(str(rel))
    if rel.is_absolute():
        candidates = [rel]
    else:
        # Roboflow exports often use ../train/images relative to a data.yaml at export root.
        candidates = [root / rel, yaml_dir / rel]
        normalized = str(rel).replace('\\', '/').removeprefix('./')
        while normalized.startswith('../'):
            normalized = normalized[3:]
        candidates += [root / normalized, yaml_dir / normalized]
        # Many Roboflow exports also use 'valid' instead of 'val'.
        for alias in (['valid'] if split == 'val' else [split]):
            candidates += [root / alias / 'images']
    for c in candidates:
        if c.is_dir() and image_list(c):
            return c.resolve()
    raise FileNotFoundError(f'No images for {split}. Tried: {[str(x) for x in candidates]}')


def inspect_dataset(dataset, out, strict=False):
    dataset = Path(dataset).expanduser().resolve()
    yaml_path = dataset if dataset.is_file() else dataset / 'data.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f'Missing {yaml_path} (export Roboflow dataset as YOLO detection format)')
    root = yaml_path.parent
    yml = read_yaml(yaml_path)
    if not isinstance(yml, dict) or 'names' not in yml:
        raise ValueError('Expected YOLO detection data.yaml with names and train/val/test')
    names = yml['names']
    if isinstance(names, list):
        name_map = dict(enumerate(names))
    else:
        name_map = {int(k): v for k, v in names.items()}
    if sorted(name_map) != list(range(len(name_map))):
        raise ValueError('Class names in data.yaml must have consecutive zero-based IDs')
    folders = {s: resolve_split(root, yaml_path.parent, yml, s) for s in ('train', 'val', 'test')}
    if not folders['train'] or not folders['val']:
        raise ValueError('train and val are required')
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path_out = out / 'data_resolved.yaml'
    yaml.safe_dump({'path': str(root), **{s: str(p) for s, p in folders.items() if p}, 'names': name_map}, path_out.open('w'), sort_keys=False, allow_unicode=True)
    records, issues, class_counts = [], [], Counter()
    split_files, seen_hashes = {}, defaultdict(list)
    groups = defaultdict(set)
    for split, folder in folders.items():
        if folder is None:
            continue
        ims = image_list(folder)
        split_files[split] = len(ims)
        for im in ims:
            # Standard Ultralytics path mapping replaces /images/ with /labels/.
            parts = list(im.parts)
            try:
                idx = len(parts) - 1 - parts[::-1].index('images')
            except ValueError:
                raise ValueError(f'Expected images/ folder in dataset: {im}')
            parts[idx] = 'labels'
            label_file = Path(*parts).with_suffix('.txt')
            labels = 0
            if label_file.exists():
                for line_num, line in enumerate(label_file.read_text(encoding='utf-8').splitlines(), 1):
                    if not line.strip():
                        continue
                    toks = line.split()
                    if len(toks) != 5:
                        issues.append(f'Label needs 5 fields: {label_file}:{line_num}')
                        continue
                    try:
                        cid = int(toks[0]); coords = [float(x) for x in toks[1:]]
                        if cid not in name_map or not all(math.isfinite(x) for x in coords):
                            raise ValueError()
                        if not all(0 <= x <= 1 for x in coords) or coords[2] <= 0 or coords[3] <= 0:
                            raise ValueError()
                        class_counts[(split, name_map[cid])] += 1
                        labels += 1
                    except ValueError:
                        issues.append(f'Invalid class/coordinates: {label_file}:{line_num}')
            # Missing label file is permitted for background images.
            file_hash = hashlib.sha256(im.read_bytes()).hexdigest()
            seen_hashes[file_hash].append((split, str(im)))
            # Heuristic grouping for video frames: strip Roboflow .rf.hash and trailing frame numbers.
            stem = re.sub(r'\.rf\.[^.]+$', '', im.stem, flags=re.I)
            candidate = re.sub(r'([_-](?:frame|img|image|f))?[_-]?\d{3,7}$', '', stem, flags=re.I)
            # Only group if a recognizable frame token was present; don't strip identifiers like X-43.
            if re.search(r'(?i)[_-](frame|img|image|f)[_-]?\d+$', stem):
                groups[candidate].add(split)
            records.append({'split': split, 'image': str(im), 'label': str(label_file), 'objects': labels})
    duplicates = [{'sha256': h, 'splits': sorted({x[0] for x in items}), 'files': [x[1] for x in items]}
                  for h, items in seen_hashes.items() if len({x[0] for x in items}) > 1]
    possible_video_leakage = [{'frame_group': k, 'splits': sorted(v)} for k, v in groups.items() if len(v) > 1]
    if not folders['test']:
        issues.append('Missing test split: validation is available, but final held-out test cannot be run.')
    if duplicates:
        issues.append(f'Found {len(duplicates)} exact image SHA-256 collisions across splits.')
    if possible_video_leakage:
        issues.append(f'Found {len(possible_video_leakage)} possible same-video frame groups across splits; inspect manually.')
    if len(class_counts) == 0:
        issues.append('No valid bounding box labels found.')
    save_json({'classes': name_map, 'split_counts': split_files, 'instances': [{'split': s, 'class': c, 'count': n} for (s,c), n in sorted(class_counts.items())], 'issues': issues, 'duplicate_images': duplicates[:100], 'potential_video_leaks': possible_video_leakage[:100]}, out / 'audit.json')
    write_rows(out / 'images.csv', records)
    print('Splits:', split_files)
    print('Classes:', name_map)
    print('Issues:', *issues, sep='\n  - ')
    print('Normalized config:', path_out)
    if issues and strict:
        raise RuntimeError('Dataset audit reported issues. See audit.json')
    if duplicates:
        raise RuntimeError('Exact duplicate images in different splits. Fix them before training!')
    return path_out


def load_experiments(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    experiments = d['experiments'] if isinstance(d, dict) else d
    if not isinstance(experiments, list) or not experiments:
        raise ValueError('experiments.json must contain a nonempty experiments array')
    if len({e['name'] for e in experiments}) != len(experiments):
        raise ValueError('Duplicate experiment names')
    return experiments


def require_gpu(device):
    import torch
    print('PyTorch:', torch.__version__, 'CUDA wheel:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. In a separate venv install PyTorch CUDA 12.6 wheels for the V100 / driver 570.86.15.')


def metric_dict(metrics):
    # Ultralytics results.box is the detection metrics object.
    box = metrics.box
    output = {
        'precision': float(box.mp),
        'recall': float(box.mr),
        'map50': float(box.map50),
        'map50_95': float(box.map),
    }
    names = metrics.names
    class_rows = []
    class_ids = list(box.ap_class_index)
    for idx, cls_id in enumerate(class_ids):
        cid = int(cls_id)
        name = names[cid] if isinstance(names, (list, dict)) else str(cid)
        class_rows.append({'class_id': cid, 'class_name': name,
                           'ap50': float(box.ap50[idx]), 'map50_95': float(box.ap[idx])})
    return output, class_rows


def train_one(experiment, data_yaml, project, epochs, device, batch, workers, seed, resume=False):
    from ultralytics import YOLO
    model_id = experiment['model']
    name = experiment['name']
    exp_dir = Path(project).resolve() / name
    best = exp_dir / 'weights' / 'best.pt'
    if best.exists() and not resume:
        raise FileExistsError(f'{best} exists. Use a different output or --skip-existing; no silent overwrite.')
    defaults = dict(
        data=str(data_yaml), epochs=epochs, imgsz=experiment.get('imgsz', 640),
        batch=batch, device=device, workers=workers, project=str(Path(project).resolve()),
        name=name, exist_ok=True, seed=seed, deterministic=True, patience=min(20, max(5, epochs//3)),
        amp=(device != 'cpu'), cache=False, optimizer=experiment.get('optimizer', 'AdamW'),
        lr0=experiment.get('lr0', 0.001), weight_decay=experiment.get('weight_decay', 0.0005),
        cos_lr=True, close_mosaic=min(10, max(1, epochs//5)),
        # Keep horizontal flip off by default: could invert handover direction and scene layout.
        fliplr=experiment.get('fliplr', 0.0), flipud=0.0,
        mosaic=experiment.get('mosaic', 0.0), mixup=experiment.get('mixup', 0.0),
        hsv_h=experiment.get('hsv_h', 0.005), hsv_s=experiment.get('hsv_s', 0.2),
        hsv_v=experiment.get('hsv_v', 0.15), translate=experiment.get('translate', 0.03),
        scale=experiment.get('scale', 0.15), degrees=experiment.get('degrees', 0.0),
        box=experiment.get('box', 7.5), cls=experiment.get('cls', 0.5),
        plots=True, verbose=True, val=True
    )
    args_path = exp_dir / 'benchmark_config.json'
    save_json({'model': model_id, **defaults}, args_path)
    model = YOLO(model_id)
    t0 = time.perf_counter()
    model.train(**defaults)
    wall = time.perf_counter() - t0
    if not best.exists():
        raise RuntimeError(f'No trained checkpoint created at {best}')
    # Run validation using best checkpoint. DO NOT use test here.
    eval_model = YOLO(str(best))
    val = eval_model.val(data=str(data_yaml), split='val', imgsz=experiment.get('imgsz', 640),
                         device=device, batch=batch, plots=True, project=str(Path(project).resolve()),
                         name=f'{name}_validation', exist_ok=True)
    metrics, per_class = metric_dict(val)
    result = {'name': name, 'checkpoint': str(best), 'model': model_id,
              'optimizer': defaults['optimizer'], 'aug': experiment.get('aug', 'custom'),
              'imgsz': defaults['imgsz'], 'batch': batch, 'wall_seconds': round(wall, 2), **metrics}
    save_json(result, exp_dir / 'benchmark_metrics.json')
    write_rows(exp_dir / 'per_class_val.csv', per_class)
    return result


def run_sweep(args):
    require_gpu(args.device)
    data_yaml = inspect_dataset(args.data, Path(args.output)/'audit')
    experiments = load_experiments(args.config)
    if args.limit:
        experiments = experiments[:args.limit]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for e in experiments:
        print('\n======', e['name'], e['model'], '======', flush=True)
        row = None
        try:
            best = out / e['name'] / 'weights' / 'best.pt'
            metrics_file = out / e['name'] / 'benchmark_metrics.json'
            if args.skip_existing and metrics_file.exists() and best.exists():
                row = json.loads(metrics_file.read_text(encoding='utf-8'))
            else:
                row = train_one(e, data_yaml, out, args.epochs, args.device, args.batch, args.workers, args.seed)
            row['status'] = 'ok'
        except Exception as exc:
            traceback.print_exc()
            row = {'name': e['name'], 'model': e['model'], 'status': 'failed', 'error': str(exc)}
        results.append(row)
        write_rows(out / 'leaderboard.csv', results, fields=['name','checkpoint','model','optimizer','aug','imgsz','batch','wall_seconds','precision','recall','map50','map50_95','status','error'])
    winners = sorted((r for r in results if r['status'] == 'ok'), key=lambda r: r.get('map50_95', 0), reverse=True)
    save_json({'ranked_on': 'validation mAP50-95 only', 'best': winners[0] if winners else None, 'all': results}, out / 'summary.json')
    print('\nBest by VALIDATION mAP50-95:', winners[0]['name'] if winners else '(none)')
    if not winners:
        raise RuntimeError('All experiments failed; inspect logs')


def run_single(args):
    require_gpu(args.device)
    data_yaml = inspect_dataset(args.data, Path(args.output)/'audit')
    experiments = {x['name']: x for x in load_experiments(args.config)}
    if args.name not in experiments:
        raise ValueError(f'Unknown experiment {args.name}. Available: {list(experiments)}')
    result = train_one(experiments[args.name], data_yaml, args.output, args.epochs, args.device, args.batch, args.workers, args.seed)
    print(json.dumps(result, indent=2))


def run_test(args):
    require_gpu(args.device)
    data_yaml = inspect_dataset(args.data, Path(args.output)/'audit')
    d = read_yaml(data_yaml)
    if not d.get('test'):
        raise ValueError('No test set in data.yaml. Keep a held-out test set and rerun.')
    from ultralytics import YOLO
    model = YOLO(str(Path(args.checkpoint).expanduser().resolve()))
    result = model.val(data=str(data_yaml), split='test', imgsz=args.imgsz,
                       batch=args.batch, device=args.device, plots=True,
                       project=str(Path(args.output).resolve()), name='final_test', exist_ok=True)
    scores, per_class = metric_dict(result)
    save_json({'checkpoint': args.checkpoint, 'split': 'test', **scores}, Path(args.output) / 'final_test_metrics.json')
    write_rows(Path(args.output) / 'final_test_per_class.csv', per_class)
    print('Held-out test metrics:', scores)


def run_predict(args):
    require_gpu(args.device)
    from ultralytics import YOLO
    model = YOLO(str(Path(args.checkpoint).expanduser().resolve()))
    results = model.predict(source=args.source, imgsz=args.imgsz, device=args.device,
                            conf=args.conf, save=True, save_txt=True, save_conf=True,
                            project=str(Path(args.output).resolve()), name='prediction', exist_ok=True,
                            stream=True)
    # stream=True ensures files/directories of arbitrary size do not fill memory.
    count = 0
    for r in results:
        count += 1
        print('Image:', Path(r.path).name, 'detections:', len(r.boxes) if r.boxes is not None else 0)
    print('Predicted images:', count)


def main():
    p = argparse.ArgumentParser(description='ForteBank YOLO26 object detection benchmark')
    subs = p.add_subparsers(dest='cmd', required=True)
    def dataset_args(q):
        q.add_argument('--data', required=True, help='Roboflow YOLO detection dataset directory or data.yaml')
        q.add_argument('--output', default='runs/detection')
    def cuda_args(q):
        q.add_argument('--device', default='0', help='0 uses CUDA_VISIBLE_DEVICES mapping, or cpu')
        q.add_argument('--batch', type=int, default=4)
    def training_args(q):
        dataset_args(q); cuda_args(q)
        q.add_argument('--config', default=DEFAULT_EXPERIMENTS)
        q.add_argument('--epochs', type=int, default=60)
        q.add_argument('--workers', type=int, default=2)
        q.add_argument('--seed', type=int, default=42)
    inspect = subs.add_parser('inspect'); dataset_args(inspect); inspect.add_argument('--strict', action='store_true')
    sweep = subs.add_parser('sweep'); training_args(sweep)
    sweep.add_argument('--limit', type=int, default=0); sweep.add_argument('--skip-existing', action='store_true')
    train = subs.add_parser('train'); training_args(train); train.add_argument('--name', required=True)
    test = subs.add_parser('test'); dataset_args(test); cuda_args(test)
    test.add_argument('--checkpoint', required=True); test.add_argument('--imgsz', type=int, default=640)
    predict = subs.add_parser('predict'); cuda_args(predict)
    predict.add_argument('--checkpoint', required=True); predict.add_argument('--source', required=True)
    predict.add_argument('--output', default='runs/detection'); predict.add_argument('--imgsz', type=int, default=640)
    predict.add_argument('--conf', type=float, default=0.25)
    args = p.parse_args()
    if args.cmd == 'inspect': inspect_dataset(args.data, args.output, args.strict)
    elif args.cmd == 'sweep': run_sweep(args)
    elif args.cmd == 'train': run_single(args)
    elif args.cmd == 'test': run_test(args)
    elif args.cmd == 'predict': run_predict(args)

if __name__ == '__main__':
    main()
