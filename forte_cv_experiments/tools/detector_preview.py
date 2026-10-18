"""YOLO-World open vocabulary preview, not ground truth; save inspection frames at given seconds."""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
OBJECT_NAMES=["person","banknotes","cash bundle","cash bag","safe door","laptop computer","money counting machine","identity document"]

p=argparse.ArgumentParser()
p.add_argument('video')
p.add_argument('--seconds',nargs='+',type=float,default=[4,8,12,16,20])
p.add_argument('--model',default='yolov8s-world.pt')
p.add_argument('--out',default='runs/detector_preview')
a=p.parse_args()
model=YOLO(a.model)
if 'world' in a.model.lower():model.set_classes(OBJECT_NAMES)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(a.video)
if not cap.isOpened():raise FileNotFoundError(a.video)
for sec in a.seconds:
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,f=cap.read()
    if not ok:continue
    pred=model.predict(f,conf=.1,verbose=False)[0]
    pth=out/f't_{sec:.1f}.jpg'
    cv2.imwrite(str(pth),pred.plot())
    print(pth,[(pred.names[int(c)],round(float(p),2)) for c,p in zip(pred.boxes.cls,pred.boxes.conf)])
cap.release()
