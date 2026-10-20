"""Ordered, resampled trajectory supervision with explicit stop and coverage costs."""
import torch
from torch.nn import functional as F
from .pixel_geometry import charbonnier, settings

DEFAULT_WEIGHTS = dict(xy=1., width=1., start=2., delta=.25, state=1.,
                       length=1., missing=1., extra=1., mask_bce=1.,
                       mask_dice=1., type=.5, last=.5, endpoints=.5)


def resample(points, count):
    # Uniform index parameterization; differentiable in predicted coordinates.
    # This is NOT arc-length matching or a learned/discrete alignment.
    return F.interpolate(points.T[None], size=count, mode='linear',
                         align_corners=True)[0].T


def trajectory_loss(result, sample, config):
    epsilon, scale = settings(config['geometry_loss'])
    count = int(config.get('alignment_points', 64))
    if count < 2:
        raise ValueError('alignment_points must be >= 2')
    size = sample['image_size']
    offsets = sample['stroke_offsets'].tolist()
    targets = [sample['points'][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    strokes = result['strokes']
    zero = strokes[0]['geometry'].sum()*0
    terms = {k: zero for k in ('xy', 'width', 'start', 'delta', 'state', 'length', 'missing', 'extra')}
    state_weights = zero.new_tensor(config.get('state_weights', [1., 5., 10.]))
    for i, target in enumerate(targets):
        # Missing strokes use the terminal predicted point as a degenerate path.
        # Full target coverage remains in the mean, so early exit cannot drop targets.
        g = strokes[i]['geometry'] if i < len(strokes) else strokes[-1]['geometry'][-1:]
        p = resample(g, count); t = resample(target[:, :3], count)
        e = (p[:, :2]-t[:, :2])*(size-1)
        terms['xy'] = terms['xy'] + charbonnier(e, epsilon).mean()/scale
        terms['width'] = terms['width'] + charbonnier((p[:, 2:3]-t[:, 2:3])*size, epsilon).mean()/scale
        terms['start'] = terms['start'] + charbonnier(e[:1], epsilon).mean()/scale
        terms['delta'] = terms['delta'] + charbonnier(e[1:]-e[:-1], epsilon).mean()/scale
        if i >= len(strokes):
            continue
        logits = strokes[i]['state_logits']
        length = len(target)
        positions = torch.arange(len(logits), device=logits.device)
        # Keep label clock explicit: a prediction cannot move its own stop target.
        labels = torch.where(positions < length-1, 0, 2 if i == len(targets)-1 else 1)
        terms['state'] = terms['state'] + F.cross_entropy(logits, labels, weight=state_weights)
        # Counterfactual survival expectation on visited steps, with a tail through
        # the configured cap. This surrogate supplies gradient when hard rollout
        # stops early; it is not an expectation over unvisited decoder states.
        prob_continue = logits.softmax(-1)[:, 0]
        survival = torch.cumprod(prob_continue, 0)
        remaining = config['rollout']['max_points']-len(logits)
        expected_length = 1+survival[:-1].sum()+survival[-1]*remaining
        terms['length'] = terms['length'] + F.smooth_l1_loss(expected_length/size, zero.new_tensor(length/size))
    for key in ('xy', 'width', 'start', 'delta', 'state', 'length'):
        terms[key] = terms[key]/len(targets)
    if len(strokes) < len(targets):
        # This actually changes the terminal state logits; a constant count penalty
        # alone would have no useful gradient through the argmax stop decision.
        terms['missing'] = (len(targets)-len(strokes))/len(targets) * F.cross_entropy(
            strokes[-1]['state_logits'][-1:], torch.full((1,),
                0 if len(strokes[-1]['geometry']) < len(targets[len(strokes)-1]) else 1,
                dtype=torch.long, device=zero.device))
    if len(strokes) > len(targets):
        end = strokes[len(targets)-1]['state_logits'][-1:]
        terms['extra'] = (len(strokes)-len(targets))/len(targets) * F.cross_entropy(
            end, torch.full((1,), 2, dtype=torch.long, device=zero.device))
    return terms


def visual_loss(predictions, sample):
    terms = {}
    masks = torch.stack([r['mask_logits'] for r in predictions])
    truth = sample['stroke_masks']
    terms['mask_bce'] = F.binary_cross_entropy_with_logits(masks, truth)
    p, t = masks.sigmoid().flatten(1), truth.flatten(1)
    terms['mask_dice'] = (1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()
    terms['type'] = F.cross_entropy(torch.stack([r['type_logits'] for r in predictions]), sample['stroke_types'])
    last = masks.new_zeros(len(predictions)); last[-1] = 1
    terms['last'] = F.binary_cross_entropy_with_logits(torch.stack([r['last_logits'] for r in predictions]), last)
    off = sample['stroke_offsets']; points = sample['points']
    endpoints = torch.cat([points[off[:-1], :2], points[off[1:]-1, :2]], -1)
    terms['endpoints'] = F.smooth_l1_loss(torch.stack([r['endpoints'] for r in predictions]), endpoints)
    return terms


def weighted_loss(terms, config):
    weights = dict(DEFAULT_WEIGHTS, **config.get('loss_weights', {}))
    return sum(weights[k]*v for k, v in terms.items())
