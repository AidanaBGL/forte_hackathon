"""Four ready-to-finetune action backbones: 2D CNN+GRU and 3D CNN."""
from __future__ import annotations
import torch
from torch import nn
from torchvision import models
from torchvision.models import video as video_models

VIDEO_WEIGHTS={
    'r3d18':video_models.R3D_18_Weights,
    'mc3_18':video_models.MC3_18_Weights,
    'r2plus1d18':video_models.R2Plus1D_18_Weights,
}


class CNN_GRU(nn.Module):
    def __init__(self,backbone:str,n_classes:int,pretrained=True,freeze_backbone=True,hidden=256):
        super().__init__()
        if backbone=='mobilenet_gru':
            net=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            net.classifier=nn.Identity()
            dim=576
        elif backbone=='resnet18_gru':
            net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            net.fc=nn.Identity()
            dim=512
        else:
            raise ValueError(backbone)
        self.encoder=net
        self.backbone_frozen=freeze_backbone
        if freeze_backbone:
            for p in self.encoder.parameters():p.requires_grad_(False)
        self.rnn=nn.GRU(dim,hidden,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Dropout(.35),nn.Linear(hidden*2,n_classes))
    def train(self,mode:bool=True):
        super().train(mode)
        if self.backbone_frozen:
            self.encoder.eval()
        return self
    def forward(self,x):
        # x[B,C,T,H,W]
        b,c,t,h,w=x.shape
        frames=x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        if self.backbone_frozen:
            with torch.no_grad():
                features=self.encoder(frames)
        else:
            features=self.encoder(frames)
        out,_=self.rnn(features.reshape(b,t,-1))
        return self.head(out.mean(dim=1))


class FrozenVideoClassifier(nn.Module):
    def __init__(self,base,n_classes:int,freeze_backbone=False):
        super().__init__()
        self.net=base
        self.freeze_backbone=freeze_backbone
        if freeze_backbone:
            for name,p in self.net.named_parameters():
                if not name.startswith('fc.'):
                    p.requires_grad_(False)
    def train(self,mode:bool=True):
        super().train(mode)
        if self.freeze_backbone:
            for mod_name,mod in self.net.named_modules():
                if mod_name!='fc':
                    # Frozen BatchNorm stats must not drift with tiny batches.
                    if isinstance(mod,(nn.BatchNorm3d,nn.BatchNorm2d)):
                        mod.eval()
        return self
    def forward(self,x):return self.net(x)


def make_model(name:str,n_classes:int,pretrained=True,freeze_backbone=None):
    if name in ('mobilenet_gru','resnet18_gru'):
        freeze = True if freeze_backbone is None else bool(freeze_backbone)
        net=CNN_GRU(name,n_classes,pretrained,freeze)
        norm='imagenet'
        return net,norm
    if name in VIDEO_WEIGHTS:
        weights=VIDEO_WEIGHTS[name].DEFAULT if pretrained else None
        fn={'r3d18':video_models.r3d_18,'mc3_18':video_models.mc3_18,
            'r2plus1d18':video_models.r2plus1d_18}[name]
        net=fn(weights=weights)
        net.fc=nn.Linear(net.fc.in_features,n_classes)
        freeze = False if freeze_backbone is None else bool(freeze_backbone)
        return FrozenVideoClassifier(net,n_classes,freeze), 'kinetics'
    raise ValueError('Unknown model '+name+'. Use mobilenet_gru, resnet18_gru, r3d18, mc3_18, r2plus1d18')


class FocalLoss(nn.Module):
    def __init__(self,gamma=2.,class_weight=None):
        super().__init__()
        self.gamma=gamma
        self.register_buffer('weight',class_weight)
    def forward(self,logits,targets):
        ce=nn.functional.cross_entropy(logits,targets,reduction='none')
        pt=nn.functional.softmax(logits,dim=1).gather(1,targets[:,None]).squeeze(1).clamp(1e-7,1-1e-7)
        loss=((1-pt)**self.gamma)*ce
        if self.weight is not None:
            loss=loss*self.weight[targets]
        return loss.mean()


def criterion_for(name,n_classes,train_labels,device,smoothing=.1,gamma=2.):
    import numpy as np
    counts=np.bincount(np.asarray(train_labels,dtype=int),minlength=n_classes)
    # sqrt inverse: avoids extreme weights on 1-shot classes.
    weights=(np.maximum(counts,1)**(-.5))
    weights=weights / weights.mean()
    weight=torch.tensor(weights,dtype=torch.float32,device=device)
    if name=='ce':return nn.CrossEntropyLoss()
    if name=='weighted_ce':return nn.CrossEntropyLoss(weight=weight)
    if name=='smooth_ce':return nn.CrossEntropyLoss(label_smoothing=smoothing)
    if name=='focal':return FocalLoss(gamma=gamma,class_weight=weight)
    raise ValueError('Unknown loss '+name)
