"""Differentiable image-only rollout; compatible with existing model checkpoints."""
import torch
from torch.nn import functional as F
from .autoregressive import encode_image, next_stroke, point_step


def select_caches(caches, indices):
    """Compact the batch axis without detaching autoregressive history."""
    return [{kind: {name: value.index_select(0, indices) for name, value in cache.items()}
             for kind, cache in layer.items()} for layer in caches]


def rollout_batch(model, images, max_strokes=None, max_points=None):
    """Decode independent images together, compacting finished paths at each point.

    Stroke ordinal and point ordinal are shared by the active batch only. An
    image that finishes a stroke waits for the other active strokes, then joins
    the next stroke. No labels or target lengths enter decoding; caches retain
    full gradients across generated points.
    """
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 1 or images.shape[-2] != images.shape[-1]:
        raise ValueError('rollout_batch requires square images [B,1,H,H]')
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
        raise ValueError('cached rollout requires dropout=0 for train/inference parity')
    ks = model.max_strokes if max_strokes is None else max_strokes
    ts = model.max_points if max_points is None else max_points
    if not 1 <= ks <= model.max_strokes or not 1 <= ts <= model.max_points:
        raise ValueError('rollout limits exceed learned positions')
    pixels, memory, sizes = encode_image(model, images)
    history = []
    owners = list(range(len(images)))
    results = [dict(strokes=[], completed=False, truncated=True,
                    truncation_reason='max_strokes_reached', used_ground_truth=False)
               for _ in owners]
    for _ in range(ks):
        z, masks, types, cond, bias = next_stroke(model, pixels, memory, sizes, history, images.shape[-1])
        history.append(z)
        point_memory = memory.detach() if model.separate_branches else memory
        if model.separate_branches:
            cond = model.condition(torch.cat([z.detach(), types.detach().softmax(-1) @ model.type_embedding], -1))
            bias = bias.detach()
        caches = [{} for _ in model.point_layers]
        previous = None
        active = list(range(len(owners)))
        geometry = [[] for _ in owners]
        logits = [[] for _ in owners]
        events = [[] for _ in owners]
        for index in range(ts):
            g, state = point_step(model, previous, index, cond, point_memory, bias, caches)
            event_tensor = state.argmax(-1)
            # One host synchronization for the active batch, rather than one per image.
            event_values = event_tensor.tolist()
            keep = []
            for row, (owner, event) in enumerate(zip(active, event_values)):
                geometry[owner].append(g[row]); logits[owner].append(state[row]); events[owner].append(event)
                if event == 0:
                    keep.append(row)
            if not keep or index == ts-1:
                break
            previous = torch.cat([g, F.one_hot(event_tensor, 3).to(g.dtype)], -1)
            if len(keep) != len(active):
                selected = torch.tensor(keep, device=images.device)
                previous = previous.index_select(0, selected)
                cond = cond.index_select(0, selected)
                point_memory = point_memory.index_select(0, selected)
                bias = bias.index_select(0, selected)
                caches = select_caches(caches, selected)
                active = [active[row] for row in keep]
        continuing = []
        for row, owner in enumerate(owners):
            result = results[owner]
            result['strokes'].append(dict(geometry=torch.stack(geometry[row]),
                state_logits=torch.stack(logits[row]), events=events[row],
                mask_logits=masks[row], type_logits=types[row]))
            event = events[row][-1]
            if event == 2:
                result.update(completed=True, truncated=False, truncation_reason=None)
            elif event == 0:
                result['truncation_reason'] = 'max_points_reached'
            else:
                continuing.append(row)
        if not continuing:
            break
        if len(continuing) != len(owners):
            selected = torch.tensor(continuing, device=images.device)
            pixels = pixels.index_select(0, selected)
            memory = memory.index_select(0, selected)
            history = [h.index_select(0, selected) for h in history]
            owners = [owners[row] for row in continuing]
    return results


def visual_predictions_batch(model, images, counts):
    """Batch the auxiliary visual graph, retaining every labeled visual slot."""
    if len(counts) != len(images) or not counts or not all(1 <= n <= model.max_strokes for n in counts):
        raise ValueError('visual supervision requires one valid stroke count per image')
    pixels, memory, sizes = encode_image(model, images)
    history = []
    owners = list(range(len(images)))
    rows = [[] for _ in owners]
    for index in range(max(counts)):
        z, masks, types, _, _ = next_stroke(model, pixels, memory, sizes, history, images.shape[-1])
        history.append(z)
        last = model.last_head(z)[:, 0]
        endpoints = model.endpoint_head(z).sigmoid()
        for row, owner in enumerate(owners):
            rows[owner].append(dict(mask_logits=masks[row], type_logits=types[row],
                last_logits=last[row], endpoints=endpoints[row]))
        keep = [row for row, owner in enumerate(owners) if counts[owner] > index+1]
        if not keep:
            break
        if len(keep) != len(owners):
            selected = torch.tensor(keep, device=images.device)
            pixels = pixels.index_select(0, selected)
            memory = memory.index_select(0, selected)
            history = [h.index_select(0, selected) for h in history]
            owners = [owners[row] for row in keep]
    return rows


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
