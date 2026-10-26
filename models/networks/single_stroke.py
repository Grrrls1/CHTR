"""Independent stage two: ResNet, visual Transformer, FPN, serial point decoder.

Unspecified paper details are explicit implementation choices: ResNet18 (random
weights), Transformer only on C5, pre-norm blocks, one FFN per serial block.
The width head and XYZ history are calligraphy extensions.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


MODE = 'single_stroke_xy_width_v1'


def spatial_position(height, width, dim, reference):
    freq = torch.exp(torch.arange(dim//4,device=reference.device,dtype=reference.dtype)*(-math.log(10000)/(dim//4)))
    y,x = torch.meshgrid(torch.arange(height,device=reference.device,dtype=reference.dtype),
                         torch.arange(width,device=reference.device,dtype=reference.dtype),indexing='ij')
    return torch.cat([(x[...,None]*freq).sin(),(x[...,None]*freq).cos(),
                      (y[...,None]*freq).sin(),(y[...,None]*freq).cos()],-1).reshape(1,height*width,dim)


class StrokeVisualEncoder(nn.Module):
    def __init__(self, dim, heads, encoder_layers, dropout):
        super().__init__()
        net = resnet18(weights=None)
        # Native grayscale input; no external downloads or ImageNet normalization.
        net.conv1 = nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)
        self.stem = nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool)
        self.stages = nn.ModuleList([net.layer1,net.layer2,net.layer3,net.layer4])
        self.lateral = nn.ModuleList([nn.Conv2d(c,dim,1) for c in (64,128,256,512)])
        layer = nn.TransformerEncoderLayer(dim,heads,dim*4,dropout,batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,encoder_layers,norm=nn.LayerNorm(dim))
        self.smooth = nn.ModuleList([nn.Conv2d(dim,dim,3,padding=1) for _ in range(4)])

    def forward(self, images):
        x = self.stem(images); features=[]
        for stage,lateral in zip(self.stages,self.lateral):
            x = stage(x); features.append(lateral(x))
        top = features[-1]; b,d,h,w = top.shape
        tokens = top.flatten(2).transpose(1,2)
        top = self.encoder(tokens+spatial_position(h,w,d,top)).transpose(1,2).reshape(b,d,h,w)
        pyramid=[None]*4;pyramid[-1]=top
        for i in range(2,-1,-1):
            pyramid[i]=features[i]+F.interpolate(pyramid[i+1],size=features[i].shape[-2:],mode='nearest')
        result=[]
        for i in (3,2,1,0):  # P5, P4, P3, P2: 8,16,32,64 for a 256 canvas.
            feature=self.smooth[i](pyramid[i]);h,w=feature.shape[-2:]
            result.append(feature.flatten(2).transpose(1,2)+spatial_position(h,w,d,feature))
        return result


class SerialPointBlock(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__()
        self.self_attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.cross=nn.ModuleList([nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True) for _ in range(4)])
        self.norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(6)])
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*4,dim))
        self.drop=nn.Dropout(dropout)

    def forward(self,x,memory,padding):
        n=x.shape[1]; causal=torch.ones(n,n,device=x.device,dtype=torch.bool).triu(1)
        q=self.norms[0](x)
        x=x+self.drop(self.self_attn(q,q,q,attn_mask=causal,key_padding_mask=padding,need_weights=False)[0])
        for i,(attn,visual) in enumerate(zip(self.cross,memory)):
            q=self.norms[i+1](x)
            x=x+self.drop(attn(q,visual,visual,need_weights=False)[0])
        return x+self.drop(self.ffn(self.norms[5](x)))


class SingleStrokeModel(nn.Module):
    def __init__(self,dim=64,heads=4,encoder_layers=2,decoder_layers=4,max_points=4096,dropout=.1):
        super().__init__()
        if dim%4 or dim%heads or encoder_layers<1 or decoder_layers!=4 or max_points<1:
            raise ValueError('require valid dim/heads, encoder_layers>=1, four decoder layers and positive cap')
        self.max_points=max_points
        self.visual=StrokeVisualEncoder(dim,heads,encoder_layers,dropout)
        self.bos=nn.Parameter(torch.zeros(1,1,dim))
        self.point_embedding=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.position=nn.Embedding(max_points+1,dim)
        self.layers=nn.ModuleList([SerialPointBlock(dim,heads,dropout) for _ in range(decoder_layers)])
        self.norm=nn.LayerNorm(dim)
        self.xy_head=nn.Linear(dim,2)
        self.width_head=nn.Linear(dim,1)
        self.valid_head=nn.Linear(dim,1)
        # Typical normalized widths are much smaller than softplus(0).
        nn.init.constant_(self.width_head.bias,-3.)

    def decode(self,memory,history,active):
        b,n,_=history.shape
        if n>self.max_points or active.shape!=(b,n) or not bool(active[:,0].all()):
            raise ValueError('invalid history, padding or point cap; no truncation')
        x=torch.cat([self.bos.expand(b,-1,-1),self.point_embedding(history)],dim=1)
        x=x+self.position(torch.arange(n+1,device=x.device))[None]
        padding=torch.cat([torch.zeros(b,1,dtype=torch.bool,device=x.device),~active],dim=1)
        for block in self.layers:x=block(x,memory,padding)
        x=self.norm(x[:,1:])
        geometry=torch.cat([self.xy_head(x).sigmoid(),F.softplus(self.width_head(x))+1e-6],-1)
        return dict(points=geometry,valid_logits=self.valid_head(x).squeeze(-1))

    def forward(self,images,history,active):
        return self.decode(self.visual(images),history,active)

    @torch.no_grad()
    def generate(self,images,first_points,max_points=None,threshold=.5):
        """Conditioned decode: retain first XYZ; discard every invalid candidate."""
        if self.training:raise ValueError('call eval() before generation')
        cap=self.max_points if max_points is None else max_points
        if not 1<=cap<=self.max_points or not 0<threshold<1:
            raise ValueError('invalid generation cap or threshold')
        if first_points.shape!=(len(images),3) or not torch.isfinite(first_points).all():
            raise ValueError('first_points must be finite Bx3 XYZ')
        if (first_points[:,:2]<0).any() or (first_points[:,:2]>1).any() or (first_points[:,2]<=0).any():
            raise ValueError('invalid normalized first point')
        memory=self.visual(images)
        live=torch.arange(len(images),device=images.device)
        points=first_points[:,None].clone();results=[None]*len(images)
        while len(live):
            visual=[m.index_select(0,live) for m in memory]
            output=self.decode(visual,points,torch.ones(points.shape[:2],device=points.device,dtype=torch.bool))
            probabilities=output['valid_logits'][:,-1].sigmoid()
            stopped=probabilities<threshold
            capped=torch.full_like(stopped,points.shape[1]>=cap)
            finished=stopped|capped
            for j in finished.nonzero(as_tuple=False).flatten().tolist():
                reason='invalid' if bool(stopped[j]) else 'point_cap'
                results[int(live[j])]=dict(points=points[j].clone(),stop_reason=reason,
                    terminated=reason=='invalid',truncated=reason=='point_cap',next_validity=float(probabilities[j]))
            keep=(~finished).nonzero(as_tuple=False).flatten()
            if not len(keep):break
            points=torch.cat([points.index_select(0,keep),output['points'][:,-1:].index_select(0,keep)],dim=1)
            live=live.index_select(0,keep)
        return results
