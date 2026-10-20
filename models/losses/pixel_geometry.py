"""Stroke-balanced pixel Charbonnier losses for XY, width, actual starts and displacement."""
import math

import torch


def settings(config):
    if config.get('type') != 'pixel_charbonnier':
        raise ValueError('geometry_loss.type must be pixel_charbonnier')
    epsilon = float(config.get('epsilon_px', .5))
    scale = float(config.get('scale_px', 10.))
    if not math.isfinite(epsilon) or not math.isfinite(scale) or epsilon <= 0 or scale <= 0:
        raise ValueError('epsilon_px and scale_px must be finite and positive')
    return epsilon, scale


def pixel_error(pred, target, image_shape):
    height, width = image_shape[-2:]
    return (pred[..., :2].float() - target[..., :2].float()) * pred.new_tensor([width-1, height-1])


def charbonnier(error, epsilon):
    square = error.square().sum(-1)
    # Algebraically sqrt(square + eps**2) - eps, without cancellation near zero.
    return square / (torch.sqrt(square + epsilon**2) + epsilon)


def geometry_losses(pred, target, valid, image_shape, config):
    epsilon, scale = settings(config)
    counts = valid.sum(-1)
    if not bool((counts > 0).all()):
        raise ValueError('Each selected stroke must contain at least one valid point')
    error = pixel_error(pred, target, image_shape)
    point_loss = charbonnier(error, epsilon) / scale
    xy = ((point_loss * valid).sum(-1) / counts).mean()
    # Width is a diameter normalized by image size (XY uses size - 1).
    width_error = (pred[..., 2:3].float() - target[..., 2:3].float()) * image_shape[-1]
    width_values = charbonnier(width_error, epsilon) / scale
    width = ((width_values * valid).sum(-1) / counts).mean()
    start = (charbonnier(error[:, 0], epsilon) / scale).mean()
    pairs = valid[:, 1:] & valid[:, :-1]
    pair_counts = pairs.sum(-1)
    has_pairs = pair_counts > 0
    if bool(has_pairs.any()):
        displacement_error = error[:, 1:] - error[:, :-1]
        delta_values = charbonnier(displacement_error, epsilon) / scale
        per_stroke = (delta_values * pairs).sum(-1) / pair_counts.clamp_min(1)
        delta = per_stroke[has_pairs].mean()
    else:
        delta = pred.sum() * 0
    return dict(xy=xy, width=width, start=start, delta=delta)


@torch.no_grad()
def geometry_metrics(pred, target, valid, image_shape):
    error = pixel_error(pred, target, image_shape)
    distance = torch.linalg.vector_norm(error, dim=-1)
    rest = valid.clone(); rest[:, 0] = False
    width_error = (pred[..., 2] - target[..., 2]).abs() * image_shape[-1]
    return dict(xy_mae_px=error[valid].abs().mean(), xy_l2_px=distance[valid].mean(),
                start_l2_px=distance[:, 0].mean(),
                other_l2_px=distance[rest].mean() if bool(rest.any()) else distance.new_tensor(0.),
                width_mae_px=width_error[valid].mean())
