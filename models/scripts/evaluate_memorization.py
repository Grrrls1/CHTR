"""Compare same-training-image teacher-forced geometry and image-only greedy trajectories."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT.parent/'reproduction/pipeline'))
from common import render,states,iou
from models.datasets.lmdb_dataset import CHTRModelDataset,collate_strokes
from models.networks.model import StrokeTrajectoryModel
from models.networks.autoregressive import generate
from models.visualization.mask_types import save_masks,write_gallery


def view_render(points,offsets,size):
    # For a truncated prediction only: impose endpoints on a temporary rendering
    # copy. The saved predicted events are NEVER altered or declared complete.
    return render(states(np.asarray(points,np.float32).copy(),np.asarray(offsets,np.int64)),
                  np.asarray(offsets,np.int64),size=size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset', help='Local LMDB directory; overrides the path saved in the checkpoint')
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=32)
    p.add_argument('--max-strokes',type=int,default=16);p.add_argument('--max-points',type=int,default=1024)
    args=p.parse_args();torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=checkpoint['config']
    model=StrokeTrajectoryModel(**config['model']);model.load_state_dict(checkpoint['model'])
    model=model.to(args.device).eval()
    dataset_path=Path(args.dataset or config['dataset']).expanduser()
    if not dataset_path.is_absolute():
        dataset_path=(Path.cwd() if args.dataset else ROOT)/dataset_path
    dataset=CHTRModelDataset(dataset_path)
    # Deterministic coverage across semantic stroke counts; every image comes from training.
    with dataset._env().begin() as txn:
        ordered=sorted(range(len(dataset)),key=lambda i:(json.loads(txn.get(dataset.ids[i].encode()))['stroke_count'],dataset.ids[i]))
    indices=[ordered[i] for i in np.linspace(0,len(ordered)-1,min(args.limit,len(ordered)),dtype=int)]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);results=[]
    with torch.inference_mode():
        for index in indices:
            sample=dataset[index];sid=sample['sample_id'];directory=output/sid;directory.mkdir(exist_ok=True)
            size=sample['image_size'];image=sample['image'][0].numpy();target=image<.5
            label=sample['points'].numpy().copy();label[:,:2]*=size-1;label[:,2]*=size
            offsets=sample['stroke_offsets'].numpy();label_image=view_render(label,offsets,size)
            batch=collate_strokes([sample]);batch={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            teacher=model(batch,condition_mode='predicted');geometry=[];state_logits=[]
            for i in range(len(offsets)-1):
                length=int(offsets[i+1]-offsets[i]);points=teacher['geometry'][i,:length].cpu().numpy().copy()
                points[:,:2]*=size-1;points[:,2]*=size;geometry.append(points)
                state_logits.append(teacher['state_logits'][i,:length].cpu().numpy())
            teacher_points=np.c_[np.concatenate(geometry),np.zeros((len(label),3),np.float32)]
            teacher_image=view_render(teacher_points,offsets,size)
            started=time.monotonic()
            prediction=generate(model,sample['image'][None].to(args.device),
                                min(args.max_strokes,model.max_strokes),min(args.max_points,model.max_points))
            seconds=time.monotonic()-started
            free_image=view_render(prediction['points'],prediction['stroke_offsets'],size)
            np.savez_compressed(directory/'prediction.npz',points=prediction['points'],
                                stroke_offsets=np.asarray(prediction['stroke_offsets'],np.int64),
                                subpath_offsets=np.asarray(prediction['subpath_offsets'],np.int64),
                                stroke_types=np.asarray(prediction['stroke_types'],np.int64),
                                stroke_masks=prediction['stroke_masks'],stroke_type_probs=prediction['stroke_type_probs'],
                                image_size=np.array(size),truncated=np.array(prediction['truncated']),
                                mode=np.array('image_only_greedy'))
            np.savez_compressed(directory/'teacher_forced.npz',geometry=np.concatenate(geometry),
                                state_logits=np.concatenate(state_logits),ground_truth_stroke_offsets=offsets,
                                stroke_masks=teacher['mask_logits'][0].sigmoid().cpu().numpy(),
                                stroke_types=teacher['type_logits'][0].argmax(-1).cpu().numpy(),
                                stroke_type_probs=teacher['type_logits'][0].softmax(-1).cpu().numpy())
            save_masks(directory,np.rint(image*255).astype(np.uint8),dataset.names,
                       sample['stroke_masks'].numpy(),sample['stroke_types'].numpy(),
                       teacher['mask_logits'][0].sigmoid().cpu().numpy(),
                       teacher['type_logits'][0].softmax(-1).cpu().numpy(),
                       prediction['stroke_masks'],prediction['stroke_type_probs'])
            cv2.imwrite(str(directory/'input.png'),np.rint(image*255).astype(np.uint8))
            for name,array in (('label',label_image),('teacher_forced',teacher_image),('free',free_image)):
                cv2.imwrite(str(directory/(name+'.png')),255-array)
            overlay=np.full((size,size,3),255,np.uint8);overlay[target]=(0,0,255)
            overlay[free_image>=128]=(255,0,0);overlay[target&(free_image>=128)]=(0,0,0)
            cv2.imwrite(str(directory/'overlay.png'),overlay)
            result=dict(sample_id=sid,character=sample['character'],training_image=True,
                gt_strokes=len(offsets)-1,predicted_strokes=len(prediction['stroke_types']),
                label_iou=iou(label_image>=128,target),teacher_forced_iou=iou(teacher_image>=128,target),
                image_only_iou=iou(free_image>=128,target),image_only_vs_label_iou=iou(free_image>=128,label_image>=128),
                completed=prediction['completed'],truncated=prediction['truncated'],
                truncation_reason=prediction['truncation_reason'],decode_seconds=seconds,
                teacher_forced_uses_ground_truth_history_and_boundaries=True,
                free_decode_uses_ground_truth=False,
                render_for_truncated_prediction_is_display_only=prediction['truncated'])
            results.append(result);(directory/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            write_gallery(output,results,dataset.names)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    summary=dict(checkpoint=args.checkpoint,dataset=str(dataset_path.resolve()),step=checkpoint['step'],samples=len(results),
                 decode_limits=dict(max_strokes=min(args.max_strokes,model.max_strokes),
                                    max_points=min(args.max_points,model.max_points)),
                 mean_teacher_forced_iou=float(np.mean([r['teacher_forced_iou'] for r in results])),
                 mean_image_only_iou=float(np.mean([r['image_only_iou'] for r in results])),
                 completed=sum(r['completed'] for r in results),truncated=sum(r['truncated'] for r in results),
                 exact_stroke_count=sum(r['gt_strokes']==r['predicted_strokes'] for r in results),
                 sampling='same-training-images, quantiles ordered by stroke count',results=results)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    # Use standard plotting for an exportable learning curve.
    metrics=Path(args.checkpoint).parent/'metrics.jsonl'
    if metrics.exists():
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=[json.loads(line) for line in metrics.open() if line.strip()]
        fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],alpha=.5,label='training loss')
        if len(rows)>50:
            means=np.convolve([r['loss'] for r in rows],np.ones(50)/50,mode='valid')
            ax.plot([r['step'] for r in rows][49:],means,label='50-step mean')
        ax.set(xlabel='Optimizer step',ylabel='Multi-task loss');ax.legend();fig.tight_layout()
        fig.savefig(output/'training_loss.png',dpi=150);plt.close(fig)
    dataset.close();print('[summary] '+json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
