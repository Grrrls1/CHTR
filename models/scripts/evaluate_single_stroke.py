"""Label-conditioned stage-two diagnostics; not complete image-only inference."""
import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.datasets.single_stroke import StrokeCatalog,SingleStrokeDataset,collate_single_strokes
from models.losses.single_stroke import single_stroke_loss
from models.training.single_stroke import local_path,load_model,to_device,signature


def pixel_points(points):
    return points * np.array([255.,255.,256.],np.float32)


def draw(image,points):
    canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for x,y,w in points:
        cv2.circle(canvas,(round(float(x)),round(float(y))),max(1,min(128,round(float(w)/2))), (60,60,220),-1)
    return canvas


def width_curve(gt,pred):
    scale=max(float(gt[:,2].max()),float(pred[:,2].max()),1.)
    def poly(points,color):
        coords=' '.join(f'{i*250/max(len(points)-1,1):.2f},{95-float(w)*90/scale:.2f}' for i,w in enumerate(points[:,2]))
        return f'<polyline fill="none" stroke="{color}" points="{coords}" />'
    return '<svg viewBox="0 0 256 100" width="256">'+poly(gt,'black')+poly(pred,'red')+'</svg>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--training-subset',action='store_true',help='Use only saved training keys; requires --split train')
    p.add_argument('--limit',type=int,default=8);p.add_argument('--decode-max-points',type=int,default=512)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.limit<1 or args.output.exists():p.error('positive limit and new output directory required')
    if args.training_subset and args.split!='train':p.error('training-subset requires --split train')
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    model,ck=load_model(local_path(args.checkpoint),device);config=ck['config'];model.eval()
    catalog=StrokeCatalog(local_path(config['dataset']),local_path(config['split_file']))
    try:
        if signature(config,catalog)!=ck['signature']:raise ValueError('checkpoint data/split signature mismatch')
        ds=SingleStrokeDataset(catalog,args.split,config['model']['max_points'])
        available=list(range(len(ds)))
        if args.training_subset:
            usage=json.loads(local_path(args.checkpoint).with_name('data_usage.json').read_text())
            positions={s:i for i,s in enumerate(ds.ids)}
            available=[]
            for sid,k in usage['train_keys']:
                ci=positions[sid]
                if not 0<=k<catalog.rows[sid]['stroke_count']:raise ValueError('invalid saved training key')
                available.append((ds.ends[ci-1] if ci else 0)+k)
        indices=[available[int(i)] for i in np.linspace(0,len(available)-1,min(args.limit,len(available)),dtype=int)]
        args.output.mkdir(parents=True);rows=[];cards=[]
        with torch.no_grad():
            for index in indices:
                s=ds[index];batch=to_device(collate_single_strokes([s]),device)
                prediction=model(batch['images'],batch['history'],batch['active'])
                _,terms=single_stroke_loss(prediction,batch,config['width_weight'],config['valid_weight'])
                free=model.generate(batch['images'],batch['history'][:,0],args.decode_max_points,args.threshold)[0]
                gt=pixel_points(s['points'].numpy());out=pixel_points(free['points'].cpu().numpy())
                tag=f'{s["sample_id"]}_stroke{s["stroke_index"]:03d}'
                image=np.rint(s['image'][0].numpy()*255).astype(np.uint8)
                cv2.imwrite(str(args.output/(tag+'_input.png')),image)
                cv2.imwrite(str(args.output/(tag+'_gt.png')),draw(image,gt))
                cv2.imwrite(str(args.output/(tag+'_pred.png')),draw(image,out))
                np.savez_compressed(args.output/(tag+'.npz'),ground_truth_xyw_px=gt,predicted_xyw_px=out,
                                    teacher_forced_next_xyw_px=pixel_points(prediction['points'][0].cpu().numpy()),
                                    teacher_forced_validity=prediction['valid_logits'][0].sigmoid().cpu().numpy())
                row=dict(sample_id=s['sample_id'],stroke_index=s['stroke_index'],character=s['character'],split=args.split,
                         gt_points=len(gt),predicted_points=len(out),point_count_error=len(out)-len(gt),
                         stop_reason=free['stop_reason'],truncated=free['truncated'],terminated=free['terminated'],
                         first_xyz_given=True,first_xyz_exact=bool(np.array_equal(out[0],gt[0])),
                         teacher_forced={k:float(v) for k,v in terms.items()})
                rows.append(row)
                cards.append(f'<article><h2>{html.escape(s["character"])} {tag}</h2>'+''.join(f'<img src="{tag}_{kind}.png" width="256" alt="{kind}">' for kind in ('input','gt','pred'))+width_curve(gt,out)+'<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></article>')
        report=dict(condition='ground-truth stroke mask and first XY/width; not image-only inference',
                    split=args.split,training_subset=args.training_subset,checkpoint=str(args.checkpoint),step=ck['step'],threshold=args.threshold,
                    decode_max_points=args.decode_max_points,samples=rows,
                    point_count_mae=float(np.mean([abs(r['point_count_error']) for r in rows])),
                    truncated=sum(r['truncated'] for r in rows),terminated=sum(r['terminated'] for r in rows))
        (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (args.output/'index.html').write_text('<meta charset="utf-8"><title>Single stroke diagnostics</title><p>'+report['condition']+'</p><p>Input / label overlay / free decode overlay. Width: black=label, red=free prediction; each curve uses normalized sequence progress.</p>'+''.join(cards))
        print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False),flush=True)
    finally:catalog.close()


if __name__=='__main__':main()
