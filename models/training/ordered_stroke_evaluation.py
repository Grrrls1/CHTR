"""Stage-one diagnostics and full image-only mask-to-trajectory validation.

All free geometric comparisons are by sequence index, not Hungarian matching.
Missing and extra strokes contribute zero to count-penalized ordered mask IoU.
Mask diagnostics are not COCO Mask AP. Frozen-stage-two cascade metrics are
reported separately, including count-penalized trajectory IoU.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from models.datasets.ordered_stroke import collate_ordered_strokes
from models.losses.ordered_stroke import ordered_stroke_loss
from models.training.ordered_stroke import to_device
from models.networks.two_stage import decode_stroke_instances
from models.training.two_stage_metrics import compare_trajectories, summarize
from models.visualization.trajectory_gallery import write_compact_gallery, write_evaluation_index


def binary_iou(a, b):
    intersection = (a & b).sum((-2, -1)).float()
    union = (a | b).sum((-2, -1)).float()
    return intersection/union.clamp_min(1)


@torch.no_grad()
def evaluate_split(model, dataset, device, config, output, indices=None, checkpoint=None, *, point_model=None, point_checkpoint=None):
    if point_model is None or point_model.training:
        raise ValueError("full-pipeline evaluation requires a frozen eval-mode point model")
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
    point_rows, titles = [], {}
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
                cascades = decode_stroke_instances(point_model,batch['images'],results,
                    config['point_decode_max_points'],config['point_validity_threshold'],config['point_batch_size'])
                for index, sample, result, cascade in zip(batch_indices, samples, results, cascades):
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
                    point_row,pred_paths=compare_trajectories(sample['trajectories_xyw_px'],cascade['strokes'])
                    point_rows.append(point_row)
                    row['point_layer']=point_row
                    row['point_stop_reasons']=[p['stop_reason'] for p in cascade['strokes']]
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    if index in selected:
                        token=hashlib.sha256(sample['sample_id'].encode()).hexdigest()[:20]
                        title=sample.get('character','')+' '+sample['sample_id']
                        titles[token]=title
                        gallery.append(token)
                        image=np.rint(sample['image'][0].numpy()*255).astype(np.uint8)
                        write_compact_gallery(output/token,title,image,sample['trajectories_xyw_px'],pred_paths,
                            gt_masks=sample['masks'].numpy(),pred_masks=result['masks'].cpu().numpy(),metadata=row,
                            extra_arrays=dict(predicted_masks=result['masks'].cpu().numpy(),
                                predicted_starts_xyw_px=result['starts'].cpu().numpy()*[255,255,256],
                                predicted_boxes_xyxy_px=result['boxes'].cpu().numpy()*255,
                                predicted_confidences=result['confidences'].cpu().numpy(),
                                point_stop_reasons=np.asarray(row['point_stop_reasons'],dtype='U16')))
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
                      stroke_layer=dict(teacher_forced=tf,free=free),point_layer=summarize(point_rows),
                      point_checkpoint=point_checkpoint,visualization_ids=gallery,
                      selection_metric='free.ordered_mask_iou', evaluation_protocol='ordered_masks_to_point_trajectories_v2',
                      geometry_alignment='sequence index; missing/extra strokes get zero ordered IoU',
                      validity_threshold=config['validity_threshold'], decode_max_strokes=config['decode_max_strokes'],
                      mask_threshold=.5, visualized_characters=len(gallery),
                      point_decode_max_points=config['point_decode_max_points'],
                      point_validity_threshold=config['point_validity_threshold'],
                      test_used_for_model_selection=False)
        (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        write_evaluation_index(output,report,titles)
        return report
    finally:
        model.train(was_training)
