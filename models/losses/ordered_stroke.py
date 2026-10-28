"""Ordered correspondence, valid-only geometry and one invalid terminal slot."""
import math
import torch
from torch.nn import functional as F

DEFAULT_WEIGHTS = dict(cls=4., mask=5., dice=5., box=5., giou=2., start=5., width=1.)


def aligned_giou(pred, target):
    lo = torch.maximum(pred[..., :2], target[..., :2])
    hi = torch.minimum(pred[..., 2:], target[..., 2:])
    intersection = (hi-lo).clamp_min(0).prod(-1)
    a = (pred[..., 2:]-pred[..., :2]).clamp_min(0).prod(-1)
    b = (target[..., 2:]-target[..., :2]).clamp_min(0).prod(-1)
    union = a+b-intersection
    enclosing = (torch.maximum(pred[..., 2:], target[..., 2:])-
                 torch.minimum(pred[..., :2], target[..., :2])).clamp_min(0).prod(-1)
    return intersection/union.clamp_min(1e-8) - (enclosing-union)/enclosing.clamp_min(1e-8)


def character_mean(values, valid):
    """Each character has equal weight regardless of its stroke count."""
    values = values.masked_fill(~valid, 0.)
    return (values.sum(1)/valid.sum(1).clamp_min(1)).mean()


def layer_terms(prediction, batch):
    valid, active = batch['valid'], batch['active']
    cls = F.binary_cross_entropy_with_logits(prediction['valid_logits'], valid.float(), reduction='none')
    terms = dict(cls=character_mean(cls, active))
    # Select valid instances BEFORE geometric arithmetic: terminal/padding labels
    # (including NaN sentinels) must not affect values or gradients.
    logits = prediction['mask_logits'][valid]
    masks = batch['masks'][valid]
    logits = F.interpolate(logits[:, None], masks.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    mask_loss = F.binary_cross_entropy_with_logits(logits, masks, reduction='none').mean((-2, -1))
    probabilities = logits.sigmoid()
    dice = 1-(2*(probabilities*masks).sum((-2, -1))+1)/(probabilities.sum((-2, -1))+masks.sum((-2, -1))+1)
    boxes, target_boxes = prediction['boxes'][valid], batch['boxes'][valid]
    starts, target_starts = prediction['starts'][valid], batch['starts'][valid]
    epsilon = .01/256.
    error = starts[:, 2]-target_starts[:, 2]
    values = dict(mask=mask_loss, dice=dice, box=(boxes-target_boxes).abs().mean(-1),
                  giou=1-aligned_giou(boxes, target_boxes),
                  start=(starts[:, :2]-target_starts[:, :2]).abs().mean(-1),
                  width=torch.sqrt(error.square()+epsilon**2)-epsilon)
    for name, value in values.items():
        padded = prediction['valid_logits'].new_zeros(valid.shape)
        padded[valid] = value
        terms[name] = character_mean(padded, valid)
    return terms


def ordered_stroke_loss(prediction, batch, weights=None, auxiliary_weight=1.):
    coefficients = dict(DEFAULT_WEIGHTS)
    if weights:
        if set(weights)-set(coefficients):
            raise ValueError('unknown stroke loss weight')
        coefficients.update(weights)
    if any(not math.isfinite(v) or v < 0 for v in [*coefficients.values(), auxiliary_weight]):
        raise ValueError('loss weights must be finite and nonnegative')
    terms = layer_terms(prediction, batch)
    total = sum(coefficients[k]*v for k, v in terms.items())
    auxiliary = prediction.get('auxiliary', [])
    aux = total*0
    if auxiliary and auxiliary_weight:
        aux = sum(sum(coefficients[k]*v for k, v in layer_terms(p, batch).items())
                  for p in auxiliary)/len(auxiliary)
    total = total+auxiliary_weight*aux
    return total, dict(terms, auxiliary=aux, loss=total)
