"""Quick no-download smoke test for Roboflow Video COCO parser and model shapes."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from data import prepare,load_manifest,RoboflowClips
from models import make_model,criterion_for


def run():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='forte_action_') as temp:
        root=Path(temp)/'data'
        for split,number in [('train',9),('valid',3),('test',3)]:
            directory=root/split
            directory.mkdir(parents=True)
            videos=[];segs=[]
            for i in range(number):
                filename=f'{split}_{i}.mp4'
                writer=cv2.VideoWriter(str(directory/filename),cv2.VideoWriter_fourcc(*'mp4v'),12,(80,80))
                if not writer.isOpened():raise RuntimeError('Cannot write MP4 test video with OpenCV')
                for t in range(15):
                    rgb=np.full((80,80,3),20,dtype=np.uint8)
                    rgb[12:30,10+i*2:33+i*2]=((i*40+40)%255,(t*10+40)%255,180)
                    rgb[0,0]=(sum(map(ord,split))%255,i,t)
                    writer.write(rgb)
                writer.release()
                videos.append({'id':i+1,'file_name':filename,'fps':12,'duration':1.25})
                segs.append({'id':i+1,'video_id':i+1,'category_id':i%3+1,
                             'start_frame':0,'end_frame':14,
                             'timestamp_start_us':0,'timestamp_end_us':1250000})
            ann={'categories':[{'id':0,'name':'object'}]+[{'id':i+1,'name':x} for i,x in enumerate(['cash_handover','cash_storage','safe_closing'])],
                 'videos':videos,'segments':segs,'images':[],'annotations':[]}
            (directory/'_annotations.coco.json').write_text(json.dumps(ann))
        info=prepare(root,Path(temp)/'manifest')
        df,_=load_manifest(Path(temp)/'manifest')
        clip=RoboflowClips(df,frames=8,size=64,split='train',aug='light',normalization='imagenet')
        x,y,index=clip[0]
        assert tuple(x.shape)==(3,8,64,64)
        for model_name in ['mobilenet_gru','resnet18_gru','r3d18','mc3_18','r2plus1d18']:
            model,_=make_model(model_name,3,pretrained=False)
            with torch.no_grad():output=model.eval()(x[None])
            assert tuple(output.shape)==(1,3)
            print(model_name,'OK')
        for loss_name in ['ce','weighted_ce','smooth_ce','focal']:
            criterion=criterion_for(loss_name,3,df[df.split=='train'].class_id.tolist(),torch.device('cpu'))
            loss=criterion(torch.tensor([[1.,2.,-1.]]),torch.tensor([1]))
            assert torch.isfinite(loss)
            print(loss_name,'OK')
    print('All smoke tests PASSED. No pretrained weights were downloaded.')


if __name__=='__main__':run()
