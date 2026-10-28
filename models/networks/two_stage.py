"""Inference adapter to the existing job605 point interface, without retraining."""
import torch


@torch.no_grad()
def recover_trajectories(stroke_model, point_model, images, max_strokes=None,
                         max_points=None, stroke_threshold=.5, point_threshold=.5,
                         point_batch_size=32):
    if stroke_model.training or point_model.training:
        raise ValueError('both stages must be in eval mode')
    if point_batch_size < 1:
        raise ValueError('point_batch_size must be positive')
    if images.shape[1:] != (1, 256, 256) or not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
        raise ValueError('expected finite Bx1x256x256 grayscale in [0,1]')
    instances = stroke_model.generate(images, max_strokes=max_strokes, threshold=stroke_threshold)
    output = [dict(instances=r, strokes=[], terminated=r['terminated'], truncated=r['truncated']) for r in instances]
    owners = [(i, k) for i, r in enumerate(instances) for k in range(len(r['starts']))]
    for offset in range(0, len(owners), point_batch_size):
        group = owners[offset:offset+point_batch_size]
        inputs = torch.stack([torch.where(instances[i]['masks'][k][None] >= .5,
                                           images[i], torch.ones_like(images[i])) for i, k in group])
        starts = torch.stack([instances[i]['starts'][k] for i, k in group])
        paths = point_model.generate(inputs, starts, max_points=max_points, threshold=point_threshold)
        for (i, _), path in zip(group, paths):
            output[i]['strokes'].append(path)
    return output
