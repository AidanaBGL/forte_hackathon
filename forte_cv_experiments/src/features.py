"""Frozen pretrained video/frame features + motion, person/open vocabulary detections + hand landmarks.

No ML model is trained here; this module writes .npz per video for subsequent sklearn training.
"""
import json, time, warnings
from pathlib import Path
import cv2
import numpy as np

OBJECT_NAMES=['person','banknotes','cash bundle','cash bag','safe door',
              'laptop computer','money counting machine','identity document']

class VisionExtractor:
    def __init__(self, encoder='none', detector='none', hands='none',
                 hand_model='', device='auto', roi='', custom_weights='', max_side=640):
        self.encoder_name=encoder
        self.detector_name=detector
        self.hands_name=hands
        self.device=device
        self.max_side=int(max_side)
        self.roi=tuple(map(float,roi.split(','))) if roi else None
        if self.roi and (len(self.roi)!=4 or not all(0<=i<=1 for i in self.roi)):
            raise ValueError('ROI must be normalized x1,y1,x2,y2 with values 0..1')
        self.torch=None
        self.model=None
        self.preprocess=None
        self.detector=None
        self.hand_landmarker=None
        if encoder!='none':
            import torch
            self.torch=torch
            self.device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
            if encoder=='resnet18':
                from torchvision.models import resnet18,ResNet18_Weights
                from torchvision.transforms import Compose,Resize,CenterCrop,Normalize
                w=ResNet18_Weights.DEFAULT
                self.model=resnet18(weights=w)
                self.model.fc=torch.nn.Identity()
                self.preprocess=w.transforms()
            elif encoder=='r3d18':
                from torchvision.models.video import r3d_18,R3D_18_Weights
                w=R3D_18_Weights.DEFAULT
                self.model=r3d_18(weights=w)
                self.model.fc=torch.nn.Identity()
                self.preprocess=w.transforms()
            else:raise ValueError('Unknown encoder: '+encoder)
            self.model.eval().to(self.device)
            print(f'Loaded frozen pretrained {encoder} on {self.device}')
        if detector!='none':
            from ultralytics import YOLO
            if detector=='coco':
                self.detector=YOLO('yolo11n.pt')
            elif detector=='world':
                # Open-vocabulary banknotes are proposals, not certified cash detections.
                self.detector=YOLO('yolov8s-world.pt')
                self.detector.set_classes(OBJECT_NAMES)
            elif detector=='custom':
                if not custom_weights: raise ValueError('Use --custom-weights path/to/best.pt')
                self.detector=YOLO(custom_weights)
            else:raise ValueError('Unknown detector: '+detector)
            print('Loaded detector:',detector)
        if hands!='none':
            if hands!='mediapipe':raise ValueError('Unknown hands option')
            if not hand_model or not Path(hand_model).exists():
                raise ValueError('MediaPipe Tasks requires --hand-model path/to/hand_landmarker.task')
            import mediapipe as mp
            self.mp=mp
            options=mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=4)
            self.hand_landmarker=mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _crop(self,im):
        if self.roi is None:return im
        h,w=im.shape[:2];x1,y1,x2,y2=self.roi
        return im[max(0,int(y1*h)):min(h,int(y2*h)),max(0,int(x1*w)):min(w,int(x2*w))]

    def read_frames(self,cap,start,end,count):
        frames=[]
        stamps=np.linspace(start,max(start,end-0.01),count,dtype=np.float64)
        for t in stamps:
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000))
            ok,frame=cap.read()
            if not ok:
                if frames: frame=frames[-1].copy()
                else: raise RuntimeError(f'Cannot decode frame at {t:.2f}s')
            crop=self._crop(frame)
            if crop.size==0: raise ValueError('ROI yields an empty crop')
            frames.append(crop)
        return frames

    @staticmethod
    def motion_features(frames):
        small=[cv2.resize(f,(160,96),interpolation=cv2.INTER_AREA) for f in frames]
        grays=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in small]
        gs=np.stack(grays).astype(np.float32)/255
        hsv=np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2HSV) for f in small]).astype(np.float32)
        hsv=hsv / np.array([179,255,255],dtype=np.float32)
        diff=np.abs(np.diff(gs,axis=0))
        diffs=diff.reshape(diff.shape[0],-1).mean(axis=1)
        flow_features=[]
        q=[]
        for j in range(len(grays)-1):
            # Dense optical flow is a motion feature, not a semantic hand action detector.
            fl=cv2.calcOpticalFlowFarneback(grays[j],grays[j+1],None,
                   pyr_scale=0.5,levels=2,winsize=13,iterations=2,poly_n=5,poly_sigma=1.1,flags=0)
            mag=np.linalg.norm(fl,axis=2)
            flow_features.append([float(mag.mean()),float(np.percentile(mag,90)),float((mag>1.0).mean())])
            H,W=mag.shape
            q.append([float(x.mean()) for x in (mag[:H//2,:W//2],mag[:H//2,W//2:],
                                               mag[H//2:,:W//2],mag[H//2:,W//2:])])
        flow_features=np.asarray(flow_features,np.float32)
        q=np.asarray(q,np.float32)
        edges=[float((cv2.Canny(g,75,150)>0).mean()) for g in grays]
        out=np.array([
            *hsv.mean(axis=(0,1,2)).tolist(),*hsv.std(axis=(0,1,2)).tolist(),
            float(gs.mean()),float(gs.std()),
            float(diffs.mean()),float(diffs.max()),float(diffs.std()),
            *flow_features.mean(axis=0).tolist(),*flow_features.max(axis=0).tolist(),
            *q.mean(axis=0).tolist(),float(np.mean(edges)),float(np.std(edges)),
            float(np.std([np.mean(g) for g in grays])),
            float(np.mean([cv2.Laplacian(g,cv2.CV_64F).var() for g in grays])/1000),
        ],np.float32)
        return out

    def embed(self,frames):
        if self.encoder_name=='none':return np.zeros((0,),np.float32)
        torch=self.torch
        with torch.inference_mode():
            if self.encoder_name=='resnet18':
                # Pretrained ImageNet features, pooled over time; does not learn movements.
                selected=[frames[i] for i in np.linspace(0,len(frames)-1,4).round().astype(int)]
                img=torch.from_numpy(np.stack([cv2.cvtColor(cv2.resize(x,(224,224)),cv2.COLOR_BGR2RGB)
                                for x in selected])).permute(0,3,1,2).float()/255.0
                inp=torch.stack([self.preprocess(i) for i in img]).to(self.device)
                z=self.model(inp).mean(dim=0)
            else:
                selected=[frames[i] for i in np.linspace(0,len(frames)-1,16).round().astype(int)]
                vid=torch.from_numpy(np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2RGB).copy()
                                                for x in selected])).permute(0,3,1,2)
                inp=self.preprocess(vid).unsqueeze(0).to(self.device)
                z=self.model(inp)[0]
            return z.float().cpu().numpy().astype(np.float32)

    def detection_features(self,frames):
        if self.detector is None:return np.zeros((0,),np.float32)
        all_scores=[];all_counts=[];people=[]
        for im in [frames[0],frames[len(frames)//2],frames[-1]]:
            res=self.detector.predict(source=im,verbose=False,conf=0.15,imgsz=640)[0]
            scores=np.zeros(len(OBJECT_NAMES),np.float32)
            counts=np.zeros(len(OBJECT_NAMES),np.float32)
            centers=[]
            if res.boxes is not None:
                names=res.names
                for cls,conf,xyxy in zip(res.boxes.cls.cpu().tolist(),res.boxes.conf.cpu().tolist(),
                                         res.boxes.xyxy.cpu().numpy()):
                    name=str(names[int(cls)]).lower()
                    if self.detector_name=='coco':
                        # Only COCO person and laptop have equivalents; NOT banknotes.
                        ind=0 if name=='person' else 5 if name=='laptop' else None
                    else:
                        ind=next((i for i,k in enumerate(OBJECT_NAMES) if k.lower()==name),None)
                    if ind is None:continue
                    scores[ind]=max(scores[ind],float(conf))
                    counts[ind]+=1
                    if ind==0:
                        x1,y1,x2,y2=xyxy
                        centers.append((float((x1+x2)/2/im.shape[1]),float((y1+y2)/2/im.shape[0])))
            all_scores.append(scores);all_counts.append(counts)
            people.append(centers)
        # Consecutive detections are not identity tracking. Movement measures nearest person center.
        movement=[]
        for a,b in zip(people,people[1:]):
            if a and b:
                movement.append(min(np.linalg.norm(np.array(x)-np.array(y)) for x in a for y in b))
        return np.r_[np.mean(all_scores,axis=0),np.mean(all_counts,axis=0),
                     [float(np.mean(movement)) if movement else 0.0]].astype(np.float32)

    def hand_features(self,frames):
        if self.hand_landmarker is None:return np.zeros((0,),np.float32)
        centers=[];counts=[]
        for im in [frames[0],frames[len(frames)//2],frames[-1]]:
            rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            # Mediapipe tasks: no face recognition / identity.
            mp_image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
            out=self.hand_landmarker.detect(mp_image)
            counts.append(len(out.hand_landmarks))
            centers.append([(float(np.mean([p.x for p in hand])),float(np.mean([p.y for p in hand])))
                            for hand in out.hand_landmarks])
        m=[]
        for a,b in zip(centers,centers[1:]):
            if a and b:m.append(min(np.linalg.norm(np.array(x)-np.array(y)) for x in a for y in b))
        return np.array([float(np.mean(counts)),float(np.max(counts)),
                         float(np.mean(m)) if m else 0.0],np.float32)

    def window_features(self,frames):
        return np.concatenate([self.motion_features(frames),self.embed(frames),
                               self.detection_features(frames),self.hand_features(frames)]).astype(np.float32)

    def process_video(self,path,window_sec=4.0,stride_sec=2.0,frames_per_window=8,max_windows=0):
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise RuntimeError('Cannot open video: '+str(path))
        fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0 or total<=0:raise RuntimeError('Invalid video metadata: '+str(path))
        dur=total/fps
        starts=list(np.arange(0,max(dur-window_sec,0)+1e-4,stride_sec,dtype=np.float64))
        if not starts:starts=[0.0]
        if starts[-1]+window_sec < dur-0.5: starts.append(max(0.0,dur-window_sec))
        if max_windows>0:starts=starts[:max_windows]
        window_indices=[];all_inds=set()
        for start in starts:
            end=min(dur,start+window_sec)
            stamps=np.linspace(start,max(start,end-0.01),max(frames_per_window,4))
            inds=[int(np.clip(round(float(t)*fps),0,total-1)) for t in stamps]
            window_indices.append(inds)
            all_inds.update(inds)
        # Sequential decoding avoids very expensive repeated random seeks in .mov.
        # Only requested frames are retained, at a configurable resolution on the longer axis.
        cache={};targets=sorted(all_inds)
        prev=-1
        for idx in targets:
            for _ in range(idx-prev-1):
                if not cap.grab():break
            if not cap.grab():
                if not cache:raise RuntimeError('Video cannot be decoded: '+str(path))
                cache[idx]=cache[max(cache)].copy()
            else:
                ok,frame=cap.retrieve()
                if not ok:
                    if not cache:raise RuntimeError('Could not retrieve frames: '+str(path))
                    frame=cache[max(cache)].copy()
                crop=self._crop(frame)
                h,w=crop.shape[:2]
                scale=min(1.0,self.max_side/max(h,w))
                if scale<1:
                    crop=cv2.resize(crop,(max(1,int(w*scale)),max(1,int(h*scale))),
                                    interpolation=cv2.INTER_AREA)
                cache[idx]=crop
            prev=idx
        cap.release()
        features=[];tt=[]
        for s,inds in zip(starts,window_indices):
            frames=[cache[i] for i in inds]
            features.append(self.window_features(frames))
            tt.append((float(s),float(min(dur,s+window_sec))))
        return np.asarray(tt,np.float32),np.vstack(features).astype(np.float32),float(dur)

    def close(self):
        if self.hand_landmarker is not None:self.hand_landmarker.close()


def save_features(manifest,feature_dir,extractor,window_sec=4.0,stride_sec=2.0,
                  frames_per_window=8,limit=0,max_windows=0,overwrite=False):
    folder=Path(feature_dir);folder.mkdir(parents=True,exist_ok=True)
    settings={'encoder':extractor.encoder_name,'detector':extractor.detector_name,'hands':extractor.hands_name,
              'roi':extractor.roi,'max_side':extractor.max_side,'window_sec':window_sec,'stride_sec':stride_sec,
              'frames_per_window':frames_per_window,'max_windows':max_windows}
    cfg=folder/'settings.json'
    if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8'))!=settings:
        raise ValueError('Features folder has different settings. Use another --features-dir or delete folder.')
    cfg.write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    start=time.perf_counter()
    n=0
    for row in manifest:
        vid=row['video_id'];dest=folder/(vid+'.npz')
        if dest.exists() and not overwrite:continue
        st=time.perf_counter()
        try:
            t,x,d=extractor.process_video(row['path'],window_sec,stride_sec,frames_per_window,max_windows)
            np.savez_compressed(dest,times=t,X=x,duration=np.float32(d))
            n+=1
            print(f'{vid:>5}: {len(t):3d} windows, {x.shape[1]:4d} dims, {time.perf_counter()-st:.1f}s',flush=True)
        except Exception as ex:
            print(f'FAILED {vid}: {type(ex).__name__}: {ex}',flush=True)
            raise
        if limit and n>=limit:break
    print(f'Extracted {n} files in {time.perf_counter()-start:.1f}s')
