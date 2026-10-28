"""Teacher-forced diagnostics and image-only ordered-mask validation.

All free geometric comparisons are by sequence index, not Hungarian matching.
Missing and extra strokes contribute zero to count-penalized ordered mask IoU.
These diagnostics are not COCO Mask AP or a full trajectory evaluation.
"""
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from models.datasets.ordered_stroke import collate_ordered_strokes
from models.losses.ordered_stroke import ordered_stroke_loss
from models.training.ordered_stroke import to_device


def binary_iou(a, b):
    intersection = (a & b).sum((-2, -1)).float()
    union = (a | b).sum((-2, -1)).float()
    return intersection/union.clamp_min(1)


def save_gallery_sample(output, sample, result):
    token = hashlib.sha256(sample['sample_id'].encode()).hexdigest()[:20]
    original = np.uint8(np.clip(sample['image'][0].numpy()*255, 0, 255))
    images = [Image.fromarray(original).convert('RGB')]
    colors = [(220, 50, 50), (40, 130, 220), (40, 170, 80), (190, 80, 210), (220, 140, 30)]
    for masks, starts in ((sample['masks'], sample['starts']), (result['masks'], result['starts'])):
        canvas = np.full((256, 256, 3), 255, np.uint8)
        for i, mask in enumerate(masks):
            canvas[mask.cpu().numpy() >= .5] = colors[i % len(colors)]
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)
        for i, start in enumerate(starts):
            x, y = (start[:2].cpu().numpy()*255).tolist()
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(0, 0, 0))
            draw.text((x+3, y), str(i+1), fill=(0, 0, 0))
        images.append(im)
    montage = Image.new('RGB', (768, 256), 'white')
    for i, im in enumerate(images):
        montage.paste(im, (i*256, 0))
    montage.save(output/(token+'.png'))
    np.savez_compressed(output/(token+'.npz'), masks=result['masks'].cpu().numpy(),
                        starts=result['starts'].cpu().numpy(), boxes=result['boxes'].cpu().numpy(),
                        confidences=result['confidences'].cpu().numpy())
    return f'<p>{html.escape(sample["sample_id"])}: original / GT / prediction</p><img src="{token}.png" width="768">'


@torch.no_grad()
def evaluate_split(model, dataset, device, config, output, indices=None, checkpoint=None):
    indices = list(range(len(dataset))) if indices is None else list(indices)
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError('evaluation requires unique, in-range samples')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('evaluation output must be empty')
    was_training = model.training
    model.eval()
    rows, sums, tf, gallery = [], {}, {}, []
    tf_strokes = tf_valid_correct = tf_terminal_correct = 0
    gallery_count = min(config.get('visualize_characters', 30), len(indices))
    selected = {indices[int(i)] for i in np.linspace(0, len(indices)-1, gallery_count, dtype=int)}
    began = time.monotonic()
    try:
        with (output/'characters.jsonl').open('w') as stream:
            for cursor in range(0, len(indices), config['eval_batch_size']):
                batch_indices = indices[cursor:cursor+config['eval_batch_size']]
                samples = [dataset[i] for i in batch_indices]
                batch = to_device(collate_ordered_strokes(samples), device)
                prediction = model(batch['images'], batch['history_masks'], batch['history_active'])
                _, terms = ordered_stroke_loss(prediction, batch, config.get('loss_weights'), config.get('auxiliary_weight', 1.))
                for k, v in terms.items():
                    tf[k] = tf.get(k, 0.)+float(v)*len(samples)
                valid_logits = prediction['valid_logits'].sigmoid() >= config['validity_threshold']
                tf_valid_correct += int((valid_logits & batch['valid']).sum())
                terminal = batch['active'] & ~batch['valid']
                tf_terminal_correct += int((~valid_logits & terminal).sum())
                valid = batch['valid']
                tf_masks = F.interpolate(prediction['mask_logits'][valid][:, None], (256, 256),
                                         mode='bilinear', align_corners=False)[:, 0] >= 0
                ious = binary_iou(tf_masks, batch['masks'][valid].bool())
                offset = 0
                for count in batch['counts'].tolist():
                    tf['mask_iou'] = tf.get('mask_iou', 0.)+float(ious[offset:offset+count].mean())
                    offset += count
                tf_strokes += int(valid.sum())
                del prediction
                results = model.generate(batch['images'], config['decode_max_strokes'], config['validity_threshold'])
                for index, sample, result in zip(batch_indices, samples, results):
                    n, m = len(sample['masks']), len(result['masks'])
                    common = min(n, m)
                    ious = binary_iou(result['masks'][:common] >= .5, sample['masks'][:common].to(device).bool())
                    row = dict(sample_id=sample['sample_id'], true_strokes=n, predicted_strokes=m,
                               stroke_count_error=m-n, stroke_count_mae=abs(m-n), count_exact=float(n == m),
                               ordered_mask_iou=float(ious.sum())/max(n, m),
                               strict_sequence_mask50=float(n == m and bool((ious >= .5).all())),
                               terminated=float(result['terminated']), truncated=float(result['truncated']),
                               empty_masks=int(((result['masks'] >= .5).flatten(1).sum(1) == 0).sum()),
                               aligned_strokes=common, stop_reason=result['stop_reason'])
                    if common:
                        error = result['starts'][:common]-sample['starts'][:common].to(device)
                        row.update(aligned_start_xy_mae_px=float(error[:, :2].abs().mean()*255),
                                   aligned_start_l2_px=float(error[:, :2].norm(dim=-1).mean()*255),
                                   aligned_start_width_mae_px=float(error[:, 2].abs().mean()*256),
                                   aligned_start_width_bias_px=float(error[:, 2].mean()*256))
                    for key in ('stroke_count_mae', 'count_exact', 'ordered_mask_iou', 'strict_sequence_mask50', 'terminated', 'truncated'):
                        sums[key] = sums.get(key, 0.)+row[key]
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    if index in selected:
                        gallery.append(save_gallery_sample(output, sample, result))
                stream.flush()
                completed = len(rows)
                elapsed = time.monotonic()-began
                print(f'[ordered-eval] {completed}/{len(indices)} ({completed/len(indices):.1%}) '
                      f'{completed/max(elapsed, 1e-9):.3f} chars/s ETA={elapsed/completed*(len(indices)-completed):.1f}s', flush=True)
        total = len(rows)
        aligned = sum(r['aligned_strokes'] for r in rows)
        free = {k: v/total for k, v in sums.items()}
        for key in ('aligned_start_xy_mae_px', 'aligned_start_l2_px', 'aligned_start_width_mae_px', 'aligned_start_width_bias_px'):
            free[key] = sum(r.get(key, 0.)*r['aligned_strokes'] for r in rows)/aligned if aligned else None
        free.update(aligned_strokes=aligned, empty_masks=sum(r['empty_masks'] for r in rows),
                    predicted_strokes=sum(r['predicted_strokes'] for r in rows))
        tf = {k: v/total for k, v in tf.items()}
        tf.update(valid_stroke_recall=tf_valid_correct/max(tf_strokes, 1), terminal_recall=tf_terminal_correct/total)
        report = dict(split=dataset.split, characters=total, strokes=tf_strokes,
                      full_split=len(indices) == len(dataset), dataset_fingerprint=dataset.catalog.fingerprint,
                      checkpoint=checkpoint, teacher_forced=tf, free=free,
                      selection_metric='free.ordered_mask_iou', evaluation_protocol='ordered_mask_index_v1',
                      geometry_alignment='sequence index; missing/extra strokes get zero ordered IoU',
                      validity_threshold=config['validity_threshold'], decode_max_strokes=config['decode_max_strokes'],
                      mask_threshold=.5, visualized_characters=len(gallery),
                      test_used_for_model_selection=False)
        (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ordered strokes</title>'
                                        '<h1>Original / GT / predicted masks</h1>'+''.join(gallery))
        return report
    finally:
        model.train(was_training)
