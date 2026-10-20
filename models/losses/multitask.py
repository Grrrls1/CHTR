"""Losses normalized over valid strokes/points; padding never supervises."""
import torch
from torch.nn import functional as F
from .pixel_geometry import geometry_losses


DEFAULT_WEIGHTS = {'xy': 1.0, 'width': 1.0, 'state': 1.0, 'mask_bce': 1.0,
                   'mask_dice': 1.0, 'type': 0.5, 'last': 0.5,
                   'endpoints': 0.5, 'delta': 0.5, 'start': 0.0}


def multitask_loss(out, batch, weights=None, state_weights=(1.0, 5.0, 10.0), geometry_config=None):
    weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
    owners, strokes = out['owners'], out['strokes']
    target = batch['points'][owners, strokes]
    valid = batch['valid_point'][owners, strokes]
    geom = out['geometry']
    pixel_terms = geometry_losses(geom, target, valid, batch['images'].shape, geometry_config) if geometry_config else None
    terms = {
        'xy': pixel_terms['xy'] if pixel_terms is not None else F.smooth_l1_loss(geom[..., :2][valid], target[..., :2][valid]),
        'width': F.smooth_l1_loss(geom[..., 2][valid], target[..., 2][valid]),
        'state': F.cross_entropy(out['state_logits'][valid], target[..., 3:][valid].argmax(-1),
                                weight=geom.new_tensor(state_weights)),
        'type': F.cross_entropy(out['type_logits'][owners, strokes], batch['types'][owners, strokes]),
        'last': F.binary_cross_entropy_with_logits(out['last_logits'][owners, strokes], batch['last'][owners, strokes]),
        'endpoints': F.smooth_l1_loss(out['endpoints'][owners, strokes], batch['endpoints'][owners, strokes]),
    }
    mask_logits = out['mask_logits'][owners, strokes]
    masks = batch['masks'][owners, strokes]
    terms['mask_bce'] = F.binary_cross_entropy_with_logits(mask_logits, masks)
    p = mask_logits.sigmoid().flatten(1)
    m = masks.flatten(1)
    terms['mask_dice'] = (1 - (2*(p*m).sum(1)+1)/(p.sum(1)+m.sum(1)+1)).mean()
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if pixel_terms is not None:
        terms['start'] = pixel_terms['start']
        terms['delta'] = pixel_terms['delta']
    elif pair_valid.any():
        terms['delta'] = F.smooth_l1_loss((geom[:, 1:, :2]-geom[:, :-1, :2])[pair_valid],
                                         (target[:, 1:, :2]-target[:, :-1, :2])[pair_valid])
    else:
        terms['delta'] = geom.sum()*0
    total = sum(weights[name]*value for name, value in terms.items())
    return total, terms
