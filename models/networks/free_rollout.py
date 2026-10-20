"""Differentiable image-only rollout; compatible with existing model checkpoints."""
import torch
from torch.nn import functional as F
from .autoregressive import encode_image, next_stroke, point_step


def rollout(model, image, max_strokes=None, max_points=None):
    """One image; argmax events control boundaries, continuous history keeps gradients.

    No target argument exists. Full BPTT; no cache/history detachment. With
    separate_branches, only visual conditions are detached, as in model.forward.
    Existing point_step has zero attention dropout, so require dropout=0.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-2] != image.shape[-1]:
        raise ValueError('rollout requires one square image [1,1,H,H]')
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
        raise ValueError('cached rollout requires dropout=0 for train/inference parity')
    ks = model.max_strokes if max_strokes is None else max_strokes
    ts = model.max_points if max_points is None else max_points
    if not 1 <= ks <= model.max_strokes or not 1 <= ts <= model.max_points:
        raise ValueError('rollout limits exceed learned positions')
    pixels, memory, sizes = encode_image(model, image)
    point_memory = memory.detach() if model.separate_branches else memory
    history, strokes = [], []
    completed = False
    reason = None
    for _ in range(ks):
        z, mask, types, cond, bias = next_stroke(model, pixels, memory, sizes, history, image.shape[-1])
        history.append(z)
        if model.separate_branches:
            # type_embedding and condition are point parameters and must keep gradients.
            cond = model.condition(torch.cat([z.detach(), types.detach().softmax(-1) @ model.type_embedding], -1))
            bias = bias.detach()
        caches = [{} for _ in model.point_layers]
        previous = None
        geometry, logits, events = [], [], []
        for index in range(ts):
            g, state = point_step(model, previous, index, cond, point_memory, bias, caches)
            event = int(state.argmax(-1).item())
            previous = torch.cat([g, F.one_hot(state.argmax(-1), 3).to(g.dtype)], -1)
            geometry.append(g[0]); logits.append(state[0]); events.append(event)
            if event != 0:
                break
        strokes.append(dict(geometry=torch.stack(geometry), state_logits=torch.stack(logits),
                            events=events, mask_logits=mask[0], type_logits=types[0]))
        if event == 2:
            completed = True
            break
        if event == 0:
            reason = 'max_points_reached'
            break
    if not completed and reason is None:
        reason = 'max_strokes_reached'
    return dict(strokes=strokes, completed=completed, truncated=not completed,
                truncation_reason=reason, used_ground_truth=False)


def visual_predictions(model, image, count):
    """Supervise all labeled visual slots, even if point rollout ended early.

    Count controls only this auxiliary loss graph, never the generated trajectory.
    No point history (predicted or labeled) enters this branch.
    """
    if not 1 <= count <= model.max_strokes:
        raise ValueError('visual supervision exceeds model stroke positions')
    pixels, memory, sizes = encode_image(model, image)
    history, rows = [], []
    for _ in range(count):
        z, mask, types, _, _ = next_stroke(model, pixels, memory, sizes, history, image.shape[-1])
        history.append(z)
        rows.append(dict(mask_logits=mask[0], type_logits=types[0],
                         last_logits=model.last_head(z)[0, 0],
                         endpoints=model.endpoint_head(z)[0].sigmoid()))
    return rows
