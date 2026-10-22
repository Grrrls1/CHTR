"""Autonomous-stop evaluation and separately labeled target-length diagnostics."""
import html
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from models.networks.hierarchical_ar import export_trajectory
from models.losses.hierarchical_ar_loss import sample_lengths,three_task_loss,geometry_metrics
from models.training.hierarchical_ar_setup import write_json

PIPELINE=Path(__file__).resolve().parents[2]/'reproduction/pipeline'
sys.path.insert(0,str(PIPELINE))
from common import render,states,iou


def to_device(sample,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in sample.items()}


def _path_length(points,offsets):
    return float(sum(np.linalg.norm(np.diff(points[a:b,:2],axis=0),axis=1).sum()
                     for a,b in zip(offsets[:-1],offsets[1:])))


@torch.inference_mode()
def evaluate(model,dataset,indices,device,decode,output=None,split_name='training_images'):
    if not indices:raise ValueError('evaluation requires at least one image')
    directory=Path(output) if output is not None else None
    if directory is not None:
        if directory.exists() and any(directory.iterdir()):raise ValueError('evaluation output must be empty')
        directory.mkdir(parents=True,exist_ok=True)
    was_training=model.training;model.eval();rows=[];sheets=[]
    try:
        for index in indices:
            sample=to_device(dataset[index],device);lengths=sample_lengths(sample);size=sample['image_size']
            # This diagnostic receives true lengths; it is never image-only performance.
            diagnostic=model(sample['image'][None],[lengths])
            _,terms=three_task_loss(diagnostic,[sample])
            metrics=geometry_metrics(diagnostic,[sample])
            del diagnostic
            if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
            started=time.monotonic()
            result=model.generate(sample['image'][None],**decode)
            if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
            seconds=time.monotonic()-started
            prediction=export_trajectory(result,size)
            points=prediction['points'];offsets=prediction['stroke_offsets']
            # endpoints imposed only on a display copy; NPZ retains genuine events.
            image=render(states(points.copy(),offsets),offsets,size=size)
            truth=sample['image'][0].cpu().numpy()<.5
            target=sample['points'].cpu().numpy().copy();target[:,:2]*=size-1;target[:,2]*=size
            gt_offsets=sample['stroke_offsets'].cpu().numpy()
            label=render(states(np.c_[target,np.zeros((len(target),3),np.float32)],gt_offsets),gt_offsets,size=size)
            generated=np.diff(offsets).tolist();count=max(len(lengths),len(generated))
            gt_pad=lengths+[0]*(count-len(lengths));pred_pad=generated+[0]*(count-len(generated))
            path=_path_length(points,offsets);gt_path=_path_length(target,gt_offsets)
            row=dict(sample_id=sample['sample_id'],character=sample['character'],split=split_name,
                image_only_iou=iou(image>=128,truth),label_iou=iou(label>=128,truth),
                gt_strokes=len(lengths),predicted_strokes=len(generated),
                gt_points=sum(lengths),predicted_points=sum(generated),
                stroke_count_error=abs(len(lengths)-len(generated)),total_point_count_error=abs(sum(lengths)-sum(generated)),
                stroke_point_count_mae=float(np.abs(np.asarray(gt_pad)-pred_pad).mean()),
                early_point_strokes=sum(p<g for p,g in zip(pred_pad,gt_pad)),
                late_point_strokes=sum(p>g for p,g in zip(pred_pad,gt_pad)),
                predicted_path_length_px=path,gt_path_length_px=gt_path,path_length_ratio=path/max(gt_path,1e-6),
                mean_step_px=path/max(1,len(points)-len(generated)),
                completed=result['completed'],truncated=result['truncated'],truncation_reason=result['truncation_reason'],
                decode_seconds=seconds,free_decode_uses_ground_truth=False,
                diagnostic_uses_ground_truth_lengths=True,
                label_length_losses={k:float(v) for k,v in terms.items()},label_length_geometry=metrics)
            rows.append(row)
            if directory is not None:
                # Dataset IDs are untrusted path components.
                sid=sample['sample_id']
                if Path(sid).name!=sid or sid in ('.','..'):raise ValueError('unsafe sample ID')
                sub=directory/sid;sub.mkdir()
                np.savez_compressed(sub/'prediction.npz',**prediction,image_size=np.array(size),
                    point_end_probs=torch.cat([s['point_end_logits'] for s in result['strokes']]).sigmoid().cpu().numpy(),
                    stroke_end_probs=result['stroke_end_logits'].sigmoid().cpu().numpy(),
                    point_ended=np.array([s['point_ended'] for s in result['strokes']]),
                    completed=np.array(result['completed']),truncated=np.array(result['truncated']),
                    truncation_reason=np.array(result['truncation_reason'] or ''),
                    mode=np.array('hierarchical_ar_image_only'),used_ground_truth=np.array(False))
                write_json(sub/'report.json',row)
                raster=np.rint(sample['image'][0].cpu().numpy()*255).astype(np.uint8)
                overlay=np.full((size,size,3),255,np.uint8);overlay[truth]=(0,0,255)
                overlay[image>=128]=(255,0,0);overlay[truth&(image>=128)]=(0,0,0)
                for name,array in [('input',raster),('label',255-label),('free',255-image),('overlay',overlay)]:
                    if not cv2.imwrite(str(sub/(name+'.png')),array):raise RuntimeError('failed to write evaluation PNG')
                strip=np.concatenate([cv2.cvtColor(raster,cv2.COLOR_GRAY2BGR),cv2.cvtColor(255-label,cv2.COLOR_GRAY2BGR),
                                      cv2.cvtColor(255-image,cv2.COLOR_GRAY2BGR),overlay],axis=1)
                caption=np.full((24,strip.shape[1],3),255,np.uint8)
                cv2.putText(caption,f'{sid} IoU={row["image_only_iou"]:.3f} strokes={len(generated)}/{len(lengths)}',
                            (4,17),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
                sheets.append(np.concatenate([caption,strip],axis=0))
    finally:
        model.train(was_training)
    notes={'heldout_images':'held-out images', 'training_images':'training-image fitting; not generalization'}
    summary=dict(split=split_name,samples=len(rows),sampling='fixed images ordered by stroke-count coverage',
        evaluation_note=notes.get(split_name,'mixed or external images; no independent-validation claim'),
        mean_image_only_iou=float(np.mean([r['image_only_iou'] for r in rows])),
        exact_stroke_count=sum(r['gt_strokes']==r['predicted_strokes'] for r in rows),
        completed=sum(r['completed'] for r in rows),truncated=sum(r['truncated'] for r in rows),
        mean_stroke_count_error=float(np.mean([r['stroke_count_error'] for r in rows])),
        mean_total_point_count_error=float(np.mean([r['total_point_count_error'] for r in rows])),
        mean_stroke_point_count_mae=float(np.mean([r['stroke_point_count_mae'] for r in rows])),
        mean_decode_seconds=float(np.mean([r['decode_seconds'] for r in rows])),
        mean_label_length_geometry={k:float(np.mean([r['label_length_geometry'][k] for r in rows])) for k in rows[0]['label_length_geometry']},
        decode=decode,rows=rows)
    if directory is not None:
        write_json(directory/'summary.json',summary)
        if len(set(sheet.shape[1] for sheet in sheets))==1:
            cv2.imwrite(str(directory/'comparison_sheet.png'),np.concatenate(sheets,axis=0))
        cards=[]
        for row in rows:
            sid=html.escape(row['sample_id'],quote=True)
            cards.append(f'<article><h2>{html.escape(row["character"])} · {sid}</h2>'
                         f'<p>IoU {row["image_only_iou"]:.3f}; strokes {row["predicted_strokes"]}/{row["gt_strokes"]}; '
                         f'points {row["predicted_points"]}/{row["gt_points"]}; truncated {row["truncated"]}</p>'
                         + ''.join(f'<figure><img src="{sid}/{name}.png"><figcaption>{name}</figcaption></figure>'
                                   for name in ('input','label','free','overlay'))+'</article>')
        (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Three-task AR evaluation</title>'
            '<style>body{font:16px sans-serif;margin:24px}figure{display:inline-block;margin:8px}img{width:224px}article{border-bottom:1px solid #ccc}</style>'
            '<h1>Image-only autonomous-stop evaluation</h1><p>'+html.escape(summary['evaluation_note'])
            +'; input / label / generated / overlay. Overlay: red=target, blue=prediction, black=overlap.</p>'
            +''.join(cards))
    return summary
