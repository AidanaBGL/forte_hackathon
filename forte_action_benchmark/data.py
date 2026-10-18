"""Roboflow Video COCO -> leakage-safe video/segment manifest and clip dataset."""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLIT_ALIASES = {'train':'train', 'training':'train', 'valid':'val', 'validation':'val', 'val':'val', 'test':'test', 'testing':'test'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
IMAGENET_MEAN = np.array([.485,.456,.406], dtype=np.float32)
IMAGENET_STD = np.array([.229,.224,.225], dtype=np.float32)
KINETICS_MEAN = np.array([.43216,.394666,.37645], dtype=np.float32)
KINETICS_STD = np.array([.22803,.22145,.21699], dtype=np.float32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _find_source_split(json_path: Path, root: Path) -> str:
    try:
        parts = json_path.relative_to(root).parts[:-1]
    except ValueError:
        parts = json_path.parts[:-1]
    for p in parts:
        if p.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[p.lower()]
    return ''


def _collect_video_paths(root: Path) -> dict[str, list[Path]]:
    d = defaultdict(list)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            d[p.name].append(p.resolve())
    return d


def load_annotations(root: str | Path) -> tuple[pd.DataFrame, list[str], dict]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    json_paths = []
    for p in root.rglob('*.json'):
        try:
            with open(p, encoding='utf-8') as f:
                obj = json.load(f)
            if isinstance(obj, dict) and 'segments' in obj and 'videos' in obj and 'categories' in obj:
                json_paths.append((p, obj))
        except (json.JSONDecodeError, UnicodeError, OSError):
            pass
    if not json_paths:
        raise ValueError('No Roboflow Video COCO JSON with videos, segments, categories found. '
                         'Expected train/_annotations.coco.json, valid/..., test/... or a single JSON in the root.')
    file_map = _collect_video_paths(root)
    rows, names, missing, warnings = [], set(), [], []
    cat_found = set()
    for ann_path, obj in json_paths:
        cats = {int(x['id']): str(x['name']).strip() for x in obj['categories']}
        videos = {int(v['id']): v for v in obj['videos']}
        source_split = _find_source_split(ann_path, root)
        for seg in obj['segments']:
            vid = videos.get(int(seg['video_id']))
            if vid is None:
                warnings.append(f'{ann_path.name}: segment {seg.get("id")} missing video_id')
                continue
            label = cats.get(int(seg['category_id']))
            if not label or label.lower() == 'object':
                warnings.append(f'Skipping generic/unknown action category {label!r} in {ann_path.name}')
                continue
            file_name = str(vid['file_name'])
            candidates = [ann_path.parent / file_name, root / file_name]
            candidates += file_map.get(Path(file_name).name, [])
            candidates = [p.resolve() for p in candidates if p.is_file()]
            if not candidates:
                missing.append(f'{ann_path}: {file_name}')
                continue
            in_folder = [p for p in candidates if ann_path.parent in p.parents or ann_path.parent == p.parent]
            resolved = in_folder[0] if in_folder else candidates[0]
            if len(set(map(str, candidates))) > 1 and not in_folder:
                warnings.append(f'Ambiguous file name {file_name}; selected {resolved}')
            fps = float(vid.get('fps') or 0)
            duration = float(vid.get('duration') or 0)
            if fps <= 0 or duration <= 0:
                cap = cv2.VideoCapture(str(resolved))
                if not cap.isOpened():
                    warnings.append(f'Cannot read video {resolved}')
                    continue
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                fps = fps if fps > 0 else float(actual_fps or 30)
                duration = duration if duration > 0 else float(frames / max(fps, 1))
            start = (float(seg['timestamp_start_us']) / 1e6 if seg.get('timestamp_start_us') is not None
                     else float(seg.get('start_frame', 0)) / fps)
            end = (float(seg['timestamp_end_us']) / 1e6 if seg.get('timestamp_end_us') is not None
                   else (float(seg['end_frame']) + 1) / fps)
            start = max(0., min(duration, start))
            end = max(start, min(duration, end))
            if end - start <= .03:
                warnings.append(f'Skipped empty segment in {resolved.name}: {label}')
                continue
            cat_found.add(label)
            rows.append({'path':str(resolved), 'video_name':resolved.name, 'label':label,
                         'start_sec':round(start, 6), 'end_sec':round(end, 6),
                         'duration_sec':round(duration, 6), 'fps':fps,
                         'original_split':source_split, 'annotation_file':str(ann_path),
                         'segment_id':str(seg.get('segment_id', seg.get('id'))),
                         'span_ratio':(end-start)/duration})
    if missing:
        raise FileNotFoundError('Roboflow annotation has missing videos. First examples: ' + '; '.join(missing[:8]))
    if not rows:
        raise ValueError('All segments are empty/missing or mapped to the generic object category.')
    frame = pd.DataFrame(rows).drop_duplicates(subset=['path','label','start_sec','end_sec']).reset_index(drop=True)
    # The same bytes must ALWAYS be in the same split. Example: X-4 vs X-47.
    hashes = {p: _sha256(Path(p)) for p in frame['path'].unique()}
    frame['group'] = frame.path.map(hashes)
    frame['segment_key'] = [f'seg_{k:05d}' for k in range(len(frame))]
    frame['class_id'] = -1
    names = sorted(cat_found)
    return frame, names, {'warnings': warnings, 'source_jsons': [str(p) for p,_ in json_paths]}


def _groups_for_split(df:pd.DataFrame) -> dict[str, set[str]]:
    return df.groupby('group')['label'].apply(lambda v: set(v.tolist())).to_dict()


def make_splits(frame:pd.DataFrame, seed:int=42, n_test:int=4,
                val_fraction:float=.20, test_names:list[str]|None=None,
                resplit_by_hash:bool=False) -> pd.DataFrame:
    df=frame.copy()
    if not 0 <= val_fraction < 1:
        raise ValueError('val_fraction must be [0,1)')
    # Check explicit Roboflow train/valid/test partitions first.
    explicit = {x for x in df.original_split.unique() if x}
    if resplit_by_hash:
        explicit = set()  # Explicit opt-in: ignore folder splits; group by SHA-256 below.
    if explicit:
        if (df.original_split == '').any():
            raise ValueError('Some annotation JSONs are in train/valid/test folders, others outside. '
                             'Place all JSONs in split folders or export a single unsplit dataset.')
        df['split'] = df.original_split
        n_per_group = df.groupby('group').split.nunique()
        if (n_per_group>1).any():
            bad=n_per_group[n_per_group>1].index.tolist()[:3]
            raise ValueError(f'IDENTICAL videos leak across Roboflow splits. SHA-256 groups: {bad}. '
                             'Move duplicates into the same split before training.')
        if 'train' not in explicit:
            raise ValueError(f'Explicit split folders {explicit} do not contain train')
        if 'val' not in explicit:
            train_groups = _groups_for_split(df[df.split=='train'])
            chosen = _select_groups(train_groups, max(1,round(len(train_groups)*val_fraction)), seed+1)
            df.loc[df.group.isin(chosen),'split']='val'
        if 'test' not in explicit:
            current = _groups_for_split(df[df.split=='train'])
            chosen = _select_groups(current, n_test, seed+2)
            df.loc[df.group.isin(chosen),'split']='test'
    else:
        groups = _groups_for_split(df)
        if len(groups)<3:
            raise ValueError('Need at least 3 unique video groups for train/val/test.')
        test_forced=set()
        if test_names:
            for g,sub in df.groupby('group'):
                if set(sub.video_name).intersection(set(test_names)):
                    test_forced.add(g)
            unknown = set(test_names)-set(df.video_name)
            if unknown:
                raise ValueError(f'Requested test videos not found: {sorted(unknown)}')
        test_set = test_forced or _select_groups(groups,min(n_test,len(groups)-2),seed)
        remaining={g:labels for g,labels in groups.items() if g not in test_set}
        n_val=max(1,round(len(remaining)*val_fraction))
        val_set=_select_groups(remaining,min(n_val,len(remaining)-1),seed+1)
        df['split']='train'
        df.loc[df.group.isin(test_set),'split']='test'
        df.loc[df.group.isin(val_set),'split']='val'
    counts=df.groupby('split').group.nunique().to_dict()
    if any(counts.get(x,0)==0 for x in ('train','val','test')):
        raise ValueError(f'Need nonempty train, val and test. Actual unique video counts: {counts}')
    train_labels=set(df[df.split=='train'].label)
    unseen=set(df.label)-train_labels
    if unseen:
        raise ValueError('Classes have no TRAIN examples after split: '+str(sorted(unseen))+'. '
                         'Move at least one independent video of each class into train, or collect more data.')
    assert (df.groupby('group').split.nunique() == 1).all()
    return df


def _select_groups(groups:dict[str,set[str]], count:int, seed:int) -> set[str]:
    """Reserve >=1 training group per class when possible; otherwise sample distinct videos."""
    rng=random.Random(seed)
    keys=list(groups)
    rng.shuffle(keys)
    count=max(0,min(count,len(keys)-1))
    freqs=Counter(c for g in keys for c in groups[g])
    chosen=[]
    # Prefer choosing samples from plentiful classes, while protecting singleton classes.
    for group in sorted(keys,key=lambda g:(min(freqs[c] for c in groups[g]),rng.random()),reverse=True):
        if len(chosen)>=count:
            break
        if all(freqs[c]>1 for c in groups[group]):
            chosen.append(group)
            for c in groups[group]: freqs[c]-=1
    return set(chosen)


def prepare(root:str|Path, output:str|Path, seed=42,n_test=4,val_fraction=.2,test_names=None,
            resplit_by_hash:bool=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    df,names,meta=load_annotations(root)
    cross_split_hashes=(df.groupby('group').original_split.nunique()>1)
    duplicated_groups=(df.groupby('group').path.nunique()>1)
    if resplit_by_hash:
        meta['warnings'].append(
            'IGNORED existing Roboflow train/valid/test folder assignment. '
            'Split all unique SHA-256 video groups from scratch; train/val/test are leakage-safe.')
        if cross_split_hashes.any():
            meta['warnings'].append(f'Found {int(cross_split_hashes.sum())} SHA-256 groups spanning source splits.')
    df=make_splits(df, seed,n_test,val_fraction,test_names,resplit_by_hash=resplit_by_hash)
    # Repeated copies of the same identical video and identical action annotation
    # should not cause overweighting, even after grouping into one split.
    duplicate_segments=int(df.duplicated(subset=['group','label','start_sec','end_sec']).sum())
    if duplicate_segments:
        meta['warnings'].append(f'Dropped {duplicate_segments} repeated annotations on byte-identical videos.')
        df=df.drop_duplicates(subset=['group','label','start_sec','end_sec']).copy().reset_index(drop=True)
    cls={name:i for i,name in enumerate(names)}
    df['class_id']=df.label.map(cls)
    df.to_csv(out/'manifest.csv',index=False)
    summary={
        'root':str(Path(root).resolve()),'seed':seed,
        'resplit_by_hash':bool(resplit_by_hash),
        'hash_groups_in_multiple_source_splits':int(cross_split_hashes.sum()),
        'identical_video_groups_with_copies':int(duplicated_groups.sum()),
        'removed_duplicate_segments':duplicate_segments,
        'class_to_idx':cls,'idx_to_class':names,'warnings':meta['warnings'],
        'source_jsons':meta['source_jsons'],
        'unique_videos_by_split':{k:int(v) for k,v in df.groupby('split').group.nunique().items()},
        'segments_by_split':{k:int(v) for k,v in df.split.value_counts().items()},
        'segments_by_label':{k:int(v) for k,v in df.label.value_counts().items()},
        'mostly_whole_video_segments':int((df.span_ratio>.8).sum()),
        'whole_video_segment_fraction':float((df.span_ratio>.8).mean()),
        'note':'Whole-video segment labels are WEAK video-level labels, not ground-truth event timecodes.'}
    with open(out/'dataset_info.json','w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    return summary


def load_manifest(path:str|Path):
    p=Path(path)
    df=pd.read_csv(p/'manifest.csv')
    with open(p/'dataset_info.json',encoding='utf-8') as f: info=json.load(f)
    return df,info


def _fit_letterbox(image:np.ndarray, size:int):
    h,w=image.shape[:2]
    scale=min(size/w,size/h)
    rw,rh=max(1,int(round(w*scale))),max(1,int(round(h*scale)))
    small=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.full((size,size,3),114,dtype=np.uint8)
    x=(size-rw)//2;y=(size-rh)//2
    canvas[y:y+rh,x:x+rw]=small
    return canvas


def read_video_clip(path:str,start:float,end:float,n_frames:int,size:int,train=False,
                    aug='none',view=0,views=1):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():raise RuntimeError(f'OpenCV failed to open {path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_end=max(0,(frame_count-1)/max(fps,1)) if frame_count else end
    end=max(start+1e-4,min(end,max_end))
    # Stratified temporal sampling over EXACT annotation span.
    edges=np.linspace(start,end,n_frames+1,dtype=np.float64)
    if train:
        times=edges[:-1]+np.random.random(n_frames)*(edges[1:]-edges[:-1])
    else:
        phase=(view+.5)/views
        times=edges[:-1]+phase*(edges[1:]-edges[:-1])
    frames=[];previous=None
    for sec in times:
        frame_idx=max(0,min(frame_count-1,int(round(sec*fps)))) if frame_count else max(0,int(sec*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
        ok,frame=cap.read()
        if not ok:
            if previous is None:
                cap.release()
                raise RuntimeError(f'Could not decode video frame at {sec:.2f}s in {path}')
            frame=previous.copy()
        else:
            previous=frame
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(_fit_letterbox(rgb,size))
    cap.release()
    arr=np.stack(frames,axis=0)  # T,H,W,C
    if train and aug != 'none':
        # Same spatial/color transform on ALL frames -- preserve temporal coherence.
        if np.random.random()<(.2 if aug=='light' else .45):
            strength=.08 if aug=='light' else .18
            gain=np.random.uniform(1-strength,1+strength)
            shift=np.random.uniform(-strength,strength)*255
            arr=np.clip(arr.astype(np.float32)*gain+shift,0,255).astype(np.uint8)
        if np.random.random()<(.2 if aug=='light' else .4):
            h,w=arr.shape[1:3]
            pad=int(size*(.035 if aug=='light' else .08))
            src=cv2.copyMakeBorder(arr[0],pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=[114,114,114])
            # Crop after padding by a shared shift; prevents object jitter over time.
            dx=np.random.randint(0,2*pad+1) if pad else 0
            dy=np.random.randint(0,2*pad+1) if pad else 0
            arr=np.stack([cv2.copyMakeBorder(f,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=[114,114,114])[dy:dy+h,dx:dx+w] for f in arr])
        if np.random.random()<(.05 if aug=='light' else .17):
            arr=np.stack([cv2.GaussianBlur(f,(3,3),.7) for f in arr])
        if np.random.random()<(.05 if aug=='light' else .2):
            t=np.random.randint(1,n_frames)
            arr[t]=arr[t-1]
        # No time reversal: reversing can change procedural correctness.
        # No horizontal flip by default: handedness, left/right device placement may matter.
    return arr


class RoboflowClips(Dataset):
    def __init__(self, df:pd.DataFrame,frames=16,size=112,split='train',views=1,
                 aug='none',normalization='kinetics',repeat_train=1):
        self.df=df[df.split==split].reset_index(drop=True)
        self.frames=frames;self.size=size;self.split=split
        self.views=views if split!='train' else repeat_train
        self.aug=aug if split=='train' else 'none'
        self.mean=KINETICS_MEAN if normalization=='kinetics' else IMAGENET_MEAN
        self.std=KINETICS_STD if normalization=='kinetics' else IMAGENET_STD
        if len(self.df)==0:raise ValueError(f'No segments for split={split}')
    def __len__(self):return len(self.df)*self.views
    def __getitem__(self,idx):
        seg_idx=idx//self.views
        view=idx%self.views
        row=self.df.iloc[seg_idx]
        frames=read_video_clip(str(row.path),float(row.start_sec),float(row.end_sec),
                               self.frames,self.size,train=self.split=='train',
                               aug=self.aug,view=view,views=self.views)
        x=(frames.astype(np.float32)/255.0-self.mean[None,None,None,:])/self.std[None,None,None,:]
        x=torch.from_numpy(x.transpose(3,0,1,2).copy())  # C,T,H,W
        return x,torch.tensor(int(row.class_id),dtype=torch.long),torch.tensor(seg_idx,dtype=torch.long)
