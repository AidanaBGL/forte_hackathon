"""Export diverse frames from train videos for Label Studio/CVAT/Roboflow annotation.
No test frames by default. This is for object box annotation, not automatic ground truth.
"""
import argparse, csv
from pathlib import Path
import cv2

p=argparse.ArgumentParser()
p.add_argument('--metadata',default='data/manifest.csv')
p.add_argument('--out',default='datasets/annotation_frames')
p.add_argument('--per-video',type=int,default=12)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with open(a.metadata,encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
written=0
for r in rows:
    if r['split']!='train' or r.get('duplicate_of',''):
        continue
    cap=cv2.VideoCapture(r['path'])
    dur=float(r['duration_sec'])
    for i in range(a.per_video):
        t=(i+0.5)*dur/a.per_video
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
        ok,im=cap.read()
        if ok:
            cv2.imwrite(str(out/f"{r['video_id']}_{i:03d}.jpg"),im)
            written+=1
    cap.release()
print('Frames exported:',written,'to',out)
