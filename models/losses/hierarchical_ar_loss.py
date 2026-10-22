"""Exactly three tasks, with original-point, stroke-balanced supervision."""
import math
import torch
from torch.nn import functional as F

DEFAULT_WEIGHTS = dict(geo=1.,point_end=1.,stroke_end=1.)


def sample_lengths(sample):
    offsets = sample['stroke_offsets'].tolist()
    lengths = [b-a for a,b in zip(offsets[:-1],offsets[1:])]
    if not lengths or offsets[0]!=0 or offsets[-1]!=len(sample['points']) or min(lengths)<1:
        raise ValueError('invalid target stroke offsets')
    return lengths


def balanced_end_loss(logits):
    """Equal positive/negative group means; singleton has only the positive group."""
    if logits.ndim != 1 or not len(logits):
        raise ValueError('end logits must be a nonempty vector')
    positive = F.softplus(-logits[-1])
    return positive if len(logits)==1 else (F.softplus(logits[:-1]).mean()+positive)/2


def three_task_loss(predictions, samples, weights=None):
    if not samples or len(predictions)!=len(samples):
        raise ValueError('requires one prediction per sample')
    weights = dict(DEFAULT_WEIGHTS,**(weights or {}))
    if set(weights)!=set(DEFAULT_WEIGHTS) or any(not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('only finite nonnegative geo/point_end/stroke_end weights are supported')
    per_image=[]
    for result,sample in zip(predictions,samples):
        lengths=sample_lengths(sample)
        if len(result['strokes'])!=len(lengths) or len(result['stroke_end_logits'])!=len(lengths):
            raise ValueError('training must cover all labeled strokes')
        geometry,point=[] ,[]
        offsets=sample['stroke_offsets'].tolist()
        for i,(stroke,n) in enumerate(zip(result['strokes'],lengths)):
            target=sample['points'][offsets[i]:offsets[i+1],:3]
            if stroke['geometry'].shape!=target.shape or stroke['point_end_logits'].shape!=(n,):
                raise ValueError('training must cover all original target points exactly')
            geometry.append(F.l1_loss(stroke['geometry'],target))
            point.append(balanced_end_loss(stroke['point_end_logits']))
        per_image.append(dict(geo=torch.stack(geometry).mean(),point_end=torch.stack(point).mean(),
                              stroke_end=balanced_end_loss(result['stroke_end_logits'])))
    terms={key:torch.stack([r[key] for r in per_image]).mean() for key in DEFAULT_WEIGHTS}
    return sum(weights[k]*terms[k] for k in terms),terms


@torch.no_grad()
def geometry_metrics(predictions,samples):
    rows=[]
    for result,sample in zip(predictions,samples):
        offsets=sample['stroke_offsets'].tolist();size=sample['image_size'];strokes=[]
        for i,stroke in enumerate(result['strokes']):
            target=sample['points'][offsets[i]:offsets[i+1],:3]
            error=(stroke['geometry'][:,:2]-target[:,:2])*(size-1)
            distance=torch.linalg.vector_norm(error,dim=-1)
            strokes.append(dict(xy_l2_px=distance.mean(),width_mae_px=((stroke['geometry'][:,2]-target[:,2])*size).abs().mean(),
                                start_l2_px=distance[0],end_l2_px=distance[-1]))
        rows.append({key:torch.stack([r[key] for r in strokes]).mean() for key in strokes[0]})
    return {key:float(torch.stack([r[key] for r in rows]).mean()) for key in rows[0]}
