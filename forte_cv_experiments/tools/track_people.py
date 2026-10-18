"""Pretrained COCO person detector + ByteTrack movement tracks; no face recognition.
Creates track CSV and annotated short video, for exploratory visualization only.
"""
import argparse,csv
from pathlib import Path
import cv2
from ultralytics import YOLO

p=argparse.ArgumentParser()
p.add_argument('video')
p.add_argument('--out',default='runs/person_tracking')
p.add_argument('--model',default='yolo11n.pt')
p.add_argument('--fps',type=float,default=3.0)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
model=YOLO(a.model)
cap=cv2.VideoCapture(a.video)
if not cap.isOpened():raise FileNotFoundError(a.video)
fps=cap.get(cv2.CAP_PROP_FPS)
count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration=count/fps
writer=None;records=[];prev_time=-1
for i in range(int(duration*a.fps)+1):
    t=min(i/a.fps,max(0,duration-0.01))
    if t<=prev_time:continue
    prev_time=t
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
    ok,frame=cap.read()
    if not ok:continue
    pred=model.track(frame,persist=True,classes=[0],tracker='bytetrack.yaml',verbose=False)[0]
    if writer is None:
        h,w=frame.shape[:2]
        writer=cv2.VideoWriter(str(out/'persons.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(w,h))
    boxes=pred.boxes
    if boxes is not None:
        b=boxes.xyxy.cpu().numpy();ids=boxes.id.int().cpu().numpy() if boxes.id is not None else [-1]*len(b)
        for box,id_ in zip(b,ids):
            x1,y1,x2,y2=map(int,box)
            records.append({'time_sec':round(t,3),'track_id':int(id_),
                    'center_x':round((x1+x2)/2/frame.shape[1],4),
                    'center_y':round((y1+y2)/2/frame.shape[0],4),
                    'x1':x1,'y1':y1,'x2':x2,'y2':y2})
            cv2.rectangle(frame,(x1,y1),(x2,y2),(45,190,110),2)
            cv2.putText(frame,f'person {id_}',(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(45,190,110),2)
    writer.write(frame)
cap.release()
if writer:writer.release()
if records:
    with open(out/'tracks.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print('Saved tracking preview to',out)
