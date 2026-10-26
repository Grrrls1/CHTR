"""Full-split free decoding and fixed-character galleries for stage two."""
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from models.datasets.single_stroke import collate_single_strokes
from models.losses.single_stroke import single_stroke_loss
from models.training.single_stroke import to_device


def visualization_ids(dataset, count=30):
    # Freeze actual image IDs, spread over stroke counts, preferring distinct labels.
    ordered = sorted(dataset.ids, key=lambda sid: (dataset.catalog.rows[sid]['stroke_count'], sid))
    seen = set(); unique = []
    for sid in ordered:
        label = dataset.catalog.rows[sid]['character']
        if label not in seen:
            seen.add(label); unique.append(sid)
    return [unique[int(i)] for i in np.linspace(0, len(unique)-1, min(count, len(unique)), dtype=int)] if count else []


def pixels(points):
    return np.asarray(points, dtype=np.float32) * np.array([255., 255., 256.], np.float32)


def raster(points):
    canvas = np.zeros((256, 256), np.uint8)
    xy = np.rint(points[:, :2]).astype(int)
    for i, (x, y) in enumerate(xy):
        radius = max(1, min(128, round(float(points[i, 2])/2)))
        cv2.circle(canvas, (int(x), int(y)), radius, 1, -1)
        if i:
            width = max(1, min(256, round(float(points[i-1:i+1, 2].mean()))))
            cv2.line(canvas, tuple(xy[i-1]), (int(x), int(y)), 1, width)
    return canvas.astype(bool)


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def resampled_errors(gt, pred):
    # Compare equal normalized sequence progress; this is not DTW or time alignment.
    n = max(len(gt), len(pred), 2)
    t = np.linspace(0, 1, n)
    def resample(p):
        return np.stack([np.interp(t, np.linspace(0, 1, len(p)), p[:, j]) for j in range(3)], -1)
    error = resample(pred)[1:] - resample(gt)[1:]
    return dict(xy_mae_px=float(np.abs(error[:, :2]).mean()),
                width_mae_px=float(np.abs(error[:, 2]).mean()),
                width_bias_px=float(error[:, 2].mean()))


def overlay(image, mask):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result[mask] = (60, 60, 220)
    return result


def width_curve(gt, pred):
    scale = max(float(gt[:, 2].max()), float(pred[:, 2].max()), 1.)
    def poly(p, color):
        coords = ' '.join(f'{i*250/max(len(p)-1,1):.2f},{95-float(w)*90/scale:.2f}' for i,w in enumerate(p[:,2]))
        return f'<polyline fill="none" stroke="{color}" points="{coords}" />'
    return '<svg viewBox="0 0 256 100" width="256">'+poly(gt, 'black')+poly(pred, 'red')+'</svg>'


@torch.no_grad()
def evaluate_split(model, dataset, device, config, output, fixed_ids=None, indices=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    indices = range(len(dataset)) if indices is None else indices
    if not len(indices):
        raise ValueError('empty evaluation split')
    fixed_ids = list(fixed_ids if fixed_ids is not None else visualization_ids(dataset, config.get('visualize_characters', 30)))
    if not set(fixed_ids) <= set(dataset.ids):
        raise ValueError('visualization IDs must belong to evaluation split')
    selected = set(fixed_ids); cards = {}; groups = {}
    totals = {}; free_totals = {}; count = terminated = truncated = exact_length = 0
    terminal_hits = terminal_count = valid_hits = valid_count = 0
    batch_size = config.get('eval_batch_size', 32)
    cap = config.get('decode_max_points', config['model']['max_points'])
    threshold = config.get('validity_threshold', .5)
    was_training = model.training
    model.eval()
    try:
        with (output/'strokes.jsonl').open('w') as log:
            for start in range(0, len(indices), batch_size):
                samples = [dataset[i] for i in indices[start:start+batch_size]]
                batch = to_device(collate_single_strokes(samples), device)
                teacher = model(batch['images'], batch['history'], batch['active'])
                _, terms = single_stroke_loss(teacher, batch, config['width_weight'], config['valid_weight'])
                for k, value in terms.items():
                    if k not in ('terminal_recall', 'valid_recall'):
                        totals[k] = totals.get(k, 0.) + float(value)*len(samples)
                positive = teacher['valid_logits'].sigmoid() >= threshold
                terminal = batch['active'] & ~batch['valid']
                terminal_hits += int((~positive & terminal).sum()); terminal_count += int(terminal.sum())
                valid_hits += int((positive & batch['valid']).sum()); valid_count += int(batch['valid'].sum())
                # Only image and first XY/width enter free decoding. No GT length or later points.
                generated = model.generate(batch['images'], batch['history'][:, 0], cap, threshold)
                for s, result in zip(samples, generated):
                    gt = pixels(s['points'].numpy())
                    pred = pixels(result['points'].cpu().numpy())
                    gt_mask = raster(gt); pred_mask = raster(pred)
                    errors = resampled_errors(gt, pred)
                    metrics = dict(trajectory_iou=iou(gt_mask, pred_mask),
                                   resampled_xy_mae_px=errors['xy_mae_px'],
                                   resampled_width_mae_px=errors['width_mae_px'],
                                   resampled_width_bias_px=errors['width_bias_px'],
                                   endpoint_l2_px=float(np.linalg.norm(pred[-1, :2]-gt[-1, :2])),
                                   point_count_mae=abs(len(pred)-len(gt)))
                    for k, value in metrics.items():free_totals[k] = free_totals.get(k, 0.) + value
                    terminated += int(result['terminated']); truncated += int(result['truncated'])
                    exact_length += int(len(pred) == len(gt)); count += 1
                    row = dict(sample_id=s['sample_id'], character=s['character'], stroke_index=s['stroke_index'],
                               gt_points=len(gt), predicted_points=len(pred), point_count_error=len(pred)-len(gt),
                               stop_reason=result['stop_reason'], first_xyz_exact=bool(np.array_equal(gt[0], pred[0])), **metrics)
                    log.write(json.dumps(row, ensure_ascii=False)+'\n')
                    if s['sample_id'] in selected:
                        sid = s['sample_id']; folder = output/sid; folder.mkdir(exist_ok=True)
                        image = np.rint(s['image'][0].numpy()*255).astype(np.uint8)
                        tag = f'stroke{s["stroke_index"]:03d}'
                        for kind, picture in [('input', image), ('gt', overlay(image, gt_mask)), ('pred', overlay(image, pred_mask))]:
                            cv2.imwrite(str(folder/f'{tag}_{kind}.png'), picture)
                        np.savez_compressed(folder/f'{tag}.npz', ground_truth_xyw_px=gt, predicted_xyw_px=pred)
                        if sid not in groups:
                            original = dataset.catalog.character_assets(sid)[0]
                            groups[sid] = [original, np.zeros_like(gt_mask), np.zeros_like(pred_mask)]
                        groups[sid][1] |= gt_mask; groups[sid][2] |= pred_mask
                        cards.setdefault(sid, []).append('<article>'+''.join(
                            f'<img width="256" src="{tag}_{kind}.png" alt="{kind}">' for kind in ('input','gt','pred'))
                            +width_curve(gt,pred)+'<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></article>')
                if count % max(batch_size, 1024//batch_size*batch_size) == 0:
                    print(json.dumps(dict(kind='evaluation_progress',split=dataset.split,strokes=count,total=len(indices))), flush=True)
        teacher_metrics = {k: v/count for k,v in totals.items()}
        teacher_metrics.update(terminal_recall=terminal_hits/max(terminal_count,1), valid_recall=valid_hits/max(valid_count,1))
        free_metrics = {k: v/count for k,v in free_totals.items()}
        free_metrics.update(terminated=terminated, truncated=truncated, terminated_rate=terminated/count,
                            truncated_rate=truncated/count, exact_point_count_rate=exact_length/count)
        summary = dict(split=dataset.split, full_split=len(indices)==len(dataset), strokes=count,
                       condition='ground-truth stroke mask and first XY/width; later points freely predicted',
                       teacher_forced=teacher_metrics, free=free_metrics,
                       free_trajectory_iou=free_metrics['trajectory_iou'],
                       decode_max_points=cap, threshold=threshold, visualization_ids=fixed_ids,
                       visualized_characters=len(cards),
                       metric_notes='IoU uses identical XY/diameter rasterization of GT and prediction. Resampled errors compare equal normalized sequence progress, exclude the given first point, and are not DTW. Stroke metrics are averaged equally; validity recalls use global point counts.')
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        links = []
        for sid in fixed_ids:
            if sid not in cards:continue
            original, gt_mask, pred_mask = groups[sid]
            folder = output/sid
            cv2.imwrite(str(folder/'original.png'), original)
            cv2.imwrite(str(folder/'gt.png'), overlay(original,gt_mask))
            cv2.imwrite(str(folder/'pred.png'), overlay(original,pred_mask))
            title = html.escape(dataset.catalog.rows[sid]['character']+' '+sid)
            (folder/'index.html').write_text('<meta charset="utf-8"><h1>'+title+'</h1><p>Input / GT / free prediction; width: black=GT, red=prediction.</p>'+''.join(cards[sid]))
            links.append(f'<section><h2><a href="{sid}/index.html">{title}</a></h2>'+''.join(
                f'<img width="256" src="{sid}/{kind}.png" alt="{kind}">' for kind in ('original','gt','pred'))+'</section>')
        (output/'index.html').write_text('<meta charset="utf-8"><title>Single stroke evaluation</title><h1>Label-conditioned free decoding</h1><pre>'
            +html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(links))
        return summary
    finally:
        model.train(was_training)
