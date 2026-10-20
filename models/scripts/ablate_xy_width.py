"""Counterfactual XY/diameter rerendering for teacher-forced trajectories."""
import argparse
import html
import io
import json
from pathlib import Path
import sys

import cv2
import lmdb
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'reproduction/pipeline'))
from common import render,iou

CASES=(('label','Label XY + Label W'),('pred_xy_pred_w','Pred XY + Pred W'),
       ('pred_xy_label_w','Pred XY + Label W'),('label_xy_pred_w','Label XY + Pred W'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--dataset',default=str(ROOT.parent/'data/chtr_mmah_lmdb_1915'))
    args=p.parse_args();folder=Path(args.eval_dir).resolve()
    original=json.loads((folder/'summary.json').read_text());results=[];cards=[];sheet=[]
    first_errors=[];other_errors=[];width_errors=[]
    with lmdb.open(args.dataset,readonly=True,lock=False,readahead=False) as env:
        with env.begin() as txn:
            for index,item in enumerate(original['results']):
                sid=item['sample_id'];directory=folder/sid;row=json.loads(txn.get(sid.encode()))
                with np.load(io.BytesIO(txn.get(row['trajectory_key'].encode())),allow_pickle=False) as z:
                    label=z['points'].copy();offsets=z['subpath_offsets'].copy();starts=z['stroke_offsets'][:-1]
                with np.load(directory/'teacher_forced.npz',allow_pickle=False) as z:pred=z['geometry'].copy()
                if pred.shape!=(len(label),3):raise ValueError('Teacher/label point alignment changed: '+sid)
                actual=np.c_[pred,label[:,3:]]
                label_width=actual.copy();label_width[:,2]=label[:,2]
                label_xy=actual.copy();label_xy[:,:2]=label[:,:2]
                target=cv2.imread(str(directory/'input.png'),cv2.IMREAD_GRAYSCALE)<128
                variants=(label,actual,label_width,label_xy);scores={};tiles=[];figures=[]
                for (key,title),points in zip(CASES,variants):
                    image=render(points,offsets,size=target.shape[0]);scores[key]=iou(image>=128,target)
                    filename='ablation_'+key+'.png';cv2.imwrite(str(directory/filename),255-image)
                    figures.append('<figure><img width="256" src="%s/%s"><figcaption>%s: %.4f</figcaption></figure>'%
                                   (sid,filename,title,scores[key]))
                    if index in (0,15,31):
                        tile=cv2.copyMakeBorder(cv2.cvtColor(255-image,cv2.COLOR_GRAY2BGR),35,0,0,0,
                                               cv2.BORDER_CONSTANT,value=(255,255,255))
                        cv2.putText(tile,title,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA)
                        cv2.putText(tile,'IoU %.3f'%scores[key],(5,30),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
                        tiles.append(tile)
                if abs(scores['pred_xy_pred_w']-item['teacher_forced_iou'])>1e-7:
                    raise ValueError('Rerender differs from original teacher-forced evaluation: '+sid)
                errors=np.linalg.norm(pred[:,:2]-label[:,:2],axis=1)
                first_errors.extend(errors[starts].tolist());mask=np.ones(len(label),bool);mask[starts]=False
                other_errors.extend(errors[mask].tolist());width_errors.extend(np.abs(pred[:,2]-label[:,2]).tolist())
                results.append(dict(sample_id=sid,character=item['character'],scores=scores,
                                    start_error_px=float(errors[starts].mean()),point_error_px=float(errors.mean()),
                                    width_mae_px=float(np.abs(pred[:,2]-label[:,2]).mean())))
                cards.append('<article><h2>'+html.escape(item['character']+' '+sid)+'</h2>'+''.join(figures)+'</article>')
                if tiles:sheet.append(np.hstack(tiles))
                if (index+1)%4==0:print('ablation %d/%d'%(index+1,len(original['results'])),flush=True)
    means={key:float(np.mean([r['scores'][key] for r in results])) for key,_ in CASES}
    summary=dict(checkpoint=original['checkpoint'],samples=len(results),mode='teacher_forced_xy_width_ablation',
                 comparison_target='normalized input image ink <128',
                 boundary_policy='ground-truth subpath boundaries shared by all four variants',
                 mean_iou=means,start_point_error_px=float(np.mean(first_errors)),
                 other_point_error_px=float(np.mean(other_errors)),width_mae_px=float(np.mean(width_errors)),
                 images_improved_by_label_xy=sum(r['scores']['label_xy_pred_w']>r['scores']['pred_xy_pred_w'] for r in results),
                 images_improved_by_label_width=sum(r['scores']['pred_xy_label_w']>r['scores']['pred_xy_pred_w'] for r in results),
                 note='Free paths have different lengths/boundaries; no pointwise GT substitution is made for free decoding.',
                 results=results)
    (folder/'width_xy_ablation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (folder/'width_xy_ablation.html').write_text('<meta charset="utf-8"><style>body{font-family:sans-serif}figure{display:inline-block;margin:5px}article{border-bottom:1px solid #aaa}</style><h1>XY / width ablation</h1><p>Teacher-forced paths. All variants share GT boundaries. IoU vs normalized input ink.</p><pre>'+html.escape(json.dumps(means,indent=2))+'</pre>'+''.join(cards))
    if sheet:cv2.imwrite(str(folder/'width_xy_ablation.png'),np.vstack(sheet))
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
