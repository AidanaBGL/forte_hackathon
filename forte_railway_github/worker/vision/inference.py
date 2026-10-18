"""ForteBank local CV pipeline. One detector pass + optional action checkpoint.

Inputs are decoded locally; inference never sends frames to an external API.
The output is a *human-review candidate*, not an automatic legal finding.
"""
import base64
import io
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .geometry import relation_events
from .procedures import classify_process, detection_events, candidate_checks, check_order_hypotheses, clean_name, rules_from_file

log = logging.getLogger(__name__)

class DeadlineExceeded(TimeoutError):
    pass


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise DeadlineExceeded('Video processing exceeded the configured deadline')


def metadata(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV cannot open video: {Path(path).name}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not 1 <= fps <= 240 or frames < 1:
        raise ValueError('Invalid video frame rate or frame count')
    return {'fps':fps,'frame_count':frames,'duration_sec':frames/fps}


def resize_letterbox(frame_bgr, target=256):
    h,w=frame_bgr.shape[:2]
    factor=min(target/w,target/h)
    rw=max(1,int(round(w*factor))); rh=max(1,int(round(h*factor)))
    small=cv2.resize(frame_bgr,(rw,rh),interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    canvas=np.full((target,target,3),114,dtype=np.uint8)
    x=(target-rw)//2;y=(target-rh)//2
    canvas[y:y+rh,x:x+rw]=small
    return cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)


def sample_video(path, det_fps=2.0, action_fps=4.0, action_size=256, deadline=None):
    """Sequential decode: avoids broken random seek on Roboflow mp4 and some MOV files.

    Returns frame tuples: detector (timestamp, BGR), action (timestamp, RGB 256 square).
    """
    m=metadata(path)
    cap=cv2.VideoCapture(str(path))
    det=[]; act=[]; idx=0; next_det=0.; next_act=0.; decoded=0
    try:
        while True:
            if deadline is not None and idx%150==0: check_deadline(deadline)
            ok,frame=cap.read()
            if not ok: break
            decoded+=1
            t=idx/m['fps']
            if t+1e-5 >= next_det:
                det.append((round(t,4),frame.copy()))
                next_det+=1.0/det_fps
            if action_fps > 0 and t+1e-5>=next_act:
                act.append((round(t,4),resize_letterbox(frame,action_size)))
                next_act+=1.0/action_fps
            idx+=1
    finally:
        cap.release()
    if decoded == 0 or not det: raise RuntimeError('The file contains no decodable frames')
    actual_duration=decoded/m['fps']
    if actual_duration < m['duration_sec']*.8:
        log.warning('Video declared %.1f sec, decoded only %.1f sec',m['duration_sec'],actual_duration)
    m['decoded_frames']=decoded
    m['decoded_duration_sec']=round(actual_duration,3)
    return m,det,act


def read_yolo_detections(model, frames, device, image_size, conf, deadline, batch=8):
    """Single local YOLO detector; model class names are looked up dynamically."""
    samples=[]
    for start in range(0,len(frames),batch):
        check_deadline(deadline)
        chunk=frames[start:start+batch]
        results=model.predict(source=[f for _,f in chunk],imgsz=image_size,conf=conf,
                              device=device,verbose=False,stream=False)
        for (t,frame),res in zip(chunk,results):
            objects=[]
            if res.boxes is not None:
                for xy,cls,p in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.cls.cpu().numpy(),res.boxes.conf.cpu().numpy()):
                    name=clean_name(res.names[int(cls)])
                    objects.append({'class_name':name, 'conf':round(float(p),3),
                                    'bbox':[round(float(v),1) for v in xy.tolist()]})
            samples.append({'t':t,'objects':objects})
    return samples


IMAGENET_MEAN=np.array([.485,.456,.406],dtype=np.float32)
IMAGENET_STD=np.array([.229,.224,.225],dtype=np.float32)
KINETICS_MEAN=np.array([.43216,.394666,.37645],dtype=np.float32)
KINETICS_STD=np.array([.22803,.22145,.216989],dtype=np.float32)


def _softmax(logits):
    z=logits-logits.max()
    e=np.exp(z)
    return e/e.sum()


def _action_meta_path(model_path):
    """Sidecar JSON written by the exporter, holding classes/frames/size/normalization."""
    p=Path(model_path)
    return p.with_name(p.stem+'.meta.json')


class _ActionBackend:
    """Shared windowing + preprocessing; subclasses only implement raw inference.

    Input to `_infer` is a float32 clip of shape [1, C, T, H, W]; it must return a
    1-D probability vector over `self.classes`.
    """
    def __init__(self,classes,frames,size,normalization):
        self.classes=list(classes)
        self.frames=int(frames)
        self.size=int(size)
        if normalization=='imagenet':
            self.mean,self.std=IMAGENET_MEAN,IMAGENET_STD
        else:
            self.mean,self.std=KINETICS_MEAN,KINETICS_STD

    def _window_input(self,frames,times,start,end):
        target=np.linspace(start+.05,end-.05,self.frames)
        indices=np.clip(np.searchsorted(times,target,side='left'),0,len(frames)-1)
        imgs=np.stack([frames[i][1] for i in indices])
        if imgs.shape[1] != self.size:
            imgs=np.stack([cv2.resize(img,(self.size,self.size)) for img in imgs])
        z=(imgs.astype(np.float32)/255.0-self.mean)/self.std
        return np.ascontiguousarray(z.transpose(3,0,1,2)[None])  # [1,C,T,H,W]

    def _infer(self,clip):  # -> np.ndarray [n_classes]
        raise NotImplementedError

    def predict_windows(self,frames,duration,window_sec,stride_sec,threshold,deadline):
        if not frames:return []
        times=np.asarray([v[0] for v in frames],dtype=float)
        positions=[];start=0.
        while start<duration:
            end=min(duration,start+window_sec)
            if end-start<.5:break
            positions.append((start,end))
            if end>=duration:break
            start+=stride_sec
        raw=[]
        for start,end in positions:
            check_deadline(deadline)
            clip=self._window_input(frames,times,start,end)
            probs=self._infer(clip)
            best=int(probs.argmax())
            raw.append({'start_sec':round(start,2),'end_sec':round(end,2),
                        'action':self.classes[best],'confidence':round(float(probs[best]),3),
                        'uncertain':bool(float(probs[best])<threshold),
                        'source':'weakly_supervised_action_model'})
        return raw


class TorchActionNet(_ActionBackend):
    """PyTorch checkpoint backend (GPU or CPU); reads metadata from the checkpoint itself."""
    def __init__(self,checkpoint,device='cpu'):
        import torch
        from .action_arch import make_model
        self.torch=torch
        self.device=torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        super().__init__(cp['classes'],cp['frames'],cp['size'],cp['normalization'])
        model,_=make_model(cp['model'],len(self.classes),pretrained=False,freeze_backbone=cp.get('freeze_backbone'))
        model.load_state_dict(cp['state_dict'],strict=True)
        self.model=model.to(self.device).eval()

    def _infer(self,clip):
        with self.torch.inference_mode():
            z=self.torch.from_numpy(clip).to(self.device)
            if self.device.type=='cuda':
                with self.torch.autocast('cuda',dtype=self.torch.float16):
                    return self.model(z).float().softmax(1)[0].cpu().numpy()
            return self.model(z).float().softmax(1)[0].cpu().numpy()


def _load_action_meta(model_path):
    meta_path=_action_meta_path(model_path)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f'Missing action metadata sidecar {meta_path.name}. Re-run the exporter '
            '(python -m worker.vision.export) to regenerate it next to the model file.')
    return json.loads(meta_path.read_text(encoding='utf-8'))


class OnnxActionNet(_ActionBackend):
    """ONNX Runtime backend — portable CPU inference, no PyTorch at runtime."""
    def __init__(self,model_path,meta=None):
        import onnxruntime as ort
        meta=meta or _load_action_meta(model_path)
        super().__init__(meta['classes'],meta['frames'],meta['size'],meta['normalization'])
        opts=ort.SessionOptions()
        threads=int(os.environ.get('FORTE_CPU_THREADS','0'))
        if threads>0:
            opts.intra_op_num_threads=threads
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input_name=self.sess.get_inputs()[0].name

    def _infer(self,clip):
        logits=self.sess.run(None,{self.input_name:clip.astype(np.float32)})[0]
        return _softmax(logits[0])


class OpenvinoActionNet(_ActionBackend):
    """OpenVINO backend — reads an IR (.xml) or an .onnx file; best CPU latency on Intel."""
    def __init__(self,model_path,meta=None):
        import openvino as ov
        meta=meta or _load_action_meta(model_path)
        super().__init__(meta['classes'],meta['frames'],meta['size'],meta['normalization'])
        core=ov.Core()
        model=core.read_model(str(model_path))
        cfg={'PERFORMANCE_HINT':'LATENCY'}
        threads=int(os.environ.get('FORTE_CPU_THREADS','0'))
        if threads>0:
            cfg['INFERENCE_NUM_THREADS']=str(threads)
        self.compiled=core.compile_model(model,'CPU',cfg)
        self.output=self.compiled.outputs[0]

    def _infer(self,clip):
        logits=self.compiled(clip.astype(np.float32))[self.output]
        return _softmax(np.asarray(logits)[0])


def load_action_model(model_path,device='cpu',backend='auto'):
    """Pick an action-recognition backend by file type (or explicit FORTE_ACTION_BACKEND).

    .pt -> PyTorch, .onnx -> ONNX Runtime, .xml -> OpenVINO IR. ONNX files can also be
    forced onto the OpenVINO runtime with FORTE_ACTION_BACKEND=openvino.
    """
    ext=Path(model_path).suffix.lower()
    if backend in (None,'','auto'):
        backend={'.pt':'torch','.pth':'torch','.onnx':'onnx','.xml':'openvino'}.get(ext)
        if backend is None:
            raise ValueError(f'Cannot infer action backend from "{ext}"; set FORTE_ACTION_BACKEND')
    if backend=='torch':
        return TorchActionNet(model_path,device)
    if backend=='onnx':
        return OnnxActionNet(model_path)
    if backend=='openvino':
        return OpenvinoActionNet(model_path)
    raise ValueError(f'Unknown FORTE_ACTION_BACKEND={backend!r} (use auto/torch/onnx/openvino)')


def combine_action_windows(windows,threshold):
    """Merge adjacent same-class windows; exact boundaries are uncalibrated candidates."""
    out=[]
    for w in windows:
        if w['confidence'] < threshold:continue
        if out and out[-1]['action']==w['action'] and out[-1]['end_sec']>=w['start_sec']-.05:
            out[-1]['end_sec']=max(out[-1]['end_sec'],w['end_sec'])
            out[-1]['confidence']=max(out[-1]['confidence'],w['confidence'])
        else:
            out.append({'action':w['action'],'start_sec':w['start_sec'],'end_sec':w['end_sec'],
                        'confidence':w['confidence'],'source':'weakly_supervised_action_model',
                        'warning':'Примерные интервалы: обучающие аннотации могли охватывать всё видео'})
    return out


def evidence_frame(path,second,max_width=960):
    cap=cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,1000*max(0,float(second)))
        ok,f=cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ok,f=cap.read()
        if not ok:raise RuntimeError('Cannot extract evidence frame')
        h,w=f.shape[:2]
        if w>max_width:f=cv2.resize(f,(max_width,int(h*max_width/w)))
        ok,jpg=cv2.imencode('.jpg',f,[int(cv2.IMWRITE_JPEG_QUALITY),80])
        if not ok:raise RuntimeError('JPEG encode failed')
        return base64.b64encode(jpg.tobytes()).decode('ascii')
    finally:cap.release()


class FortePipeline:
    def __init__(self,settings):
        settings.assert_ready()
        self.cfg=settings
        from ultralytics import YOLO
        self.detector=YOLO(str(settings.detector_weights))
        self.action=None
        if settings.action_checkpoint:
            self.action=load_action_model(settings.action_checkpoint,settings.device,settings.action_backend)
        self.rules=rules_from_file(settings.rules_path)
        self.graphs=json.loads((Path(__file__).resolve().parents[1]/'config'/'procedure_graphs.json').read_text(encoding='utf-8'))
        self.cls_names={clean_name(v) for v in self.detector.names.values()}
        log.info('Loaded detector %s with %s',settings.detector_weights,sorted(self.cls_names))

    def analyze(self,path,on_stage=None):
        def stage(msg):
            if on_stage:on_stage(msg)
        started=time.monotonic()
        deadline=started+self.cfg.job_timeout_sec
        stage('Декодирование видео')
        m,det_frames,act_frames=sample_video(path,self.cfg.sample_fps,4.0 if self.action else 0,
                                             256,deadline)
        if m['duration_sec']>self.cfg.max_duration_sec:
            raise ValueError(f'Video duration exceeds {self.cfg.max_duration_sec} seconds')
        stage('Детекция объектов')
        samples=read_yolo_detections(self.detector,det_frames,self.cfg.device,self.cfg.detector_imgsz,
                                     self.cfg.detector_conf,deadline)
        proc=classify_process(samples)
        stage('Определение действий')
        state_events=detection_events(samples)
        spatial_events=relation_events(samples)
        action_raw=[]; action_events=[]
        if self.action:
            action_raw=self.action.predict_windows(act_frames,m['decoded_duration_sec'],
                self.cfg.action_window_sec,self.cfg.action_stride_sec,self.cfg.action_threshold,deadline)
            action_events=combine_action_windows(action_raw,self.cfg.action_threshold)
        events=sorted(state_events+action_events+spatial_events,key=lambda e:e['start_sec'])
        stage('Проверка процедуры')
        warnings=candidate_checks(proc,samples,events,self.rules)
        warnings+=check_order_hypotheses(proc,events,self.graphs)
        violations=[]
        for w in warnings:
            check_deadline(deadline)
            t=w['timecode_sec']
            rule_ref='Гипотеза о рабочем процессе; подтверждённый нормативный пункт отсутствует'
            description=f'{w["description"]} Источник: {rule_ref}. Вывод требует проверки человеком.'
            violations.append({'type':w['type'],'timecode_sec':t,'description':description,
                'evidence_frames':[{'timestamp_sec':t,'image_base64':evidence_frame(path,t)}]})
        counts=Counter(o['class_name'] for s in samples for o in s['objects'])
        detail={'process':proc,'video':m,'events':events,'object_counts':dict(counts),
                'candidate_warnings':warnings,'action_windows':action_raw,
                'observations':samples,'model_info':{
                    'detector':str(Path(self.cfg.detector_weights).name),
                    'action':str(Path(self.cfg.action_checkpoint).name) if self.action else None,
                    'warning':'Action timecodes are approximate when trained on full-video labels.'},
                'processing_sec':round(time.monotonic()-started,2)}
        return {'verdict':'violation' if violations else 'no_violation','violations':violations},detail
