"""Equal-stroke means; width Charbonnier in normalized 256-pixel units."""
import torch
from torch.nn import functional as F


def masked_mean_per_stroke(values,mask):
    return ((values*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


def single_stroke_loss(output,batch,width_weight=1.,valid_weight=.2):
    if width_weight<=0 or valid_weight<=0:raise ValueError('loss weights must be positive')
    target=batch['targets'];valid=batch['valid'];active=batch['active']
    xy=masked_mean_per_stroke((output['points'][...,:2]-target[...,:2]).abs().mean(-1),valid)
    error=output['points'][...,2]-target[...,2]
    eps=.01/256.
    width=masked_mean_per_stroke(torch.sqrt(error.square()+eps**2)-eps,valid)
    validity=masked_mean_per_stroke(F.binary_cross_entropy_with_logits(output['valid_logits'],valid.float(),reduction='none'),active)
    total=xy+width_weight*width+valid_weight*validity
    with torch.no_grad():
        metrics=dict(xy_mae_px=masked_mean_per_stroke((output['points'][...,:2]-target[...,:2]).abs().mean(-1)*255,valid),
                     width_mae_px=masked_mean_per_stroke(error.abs()*256,valid),
                     width_bias_px=masked_mean_per_stroke(error*256,valid),
                     terminal_recall=((output['valid_logits'].sigmoid()<.5)&active&~valid).sum()/(active&~valid).sum().clamp_min(1),
                     valid_recall=((output['valid_logits'].sigmoid()>=.5)&valid).sum()/valid.sum().clamp_min(1))
    return total,dict(loss=total,xy=xy,width=width,validity=validity,**metrics)
