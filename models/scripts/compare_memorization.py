"""Compare fixed-image evaluations using pixel errors and free-generation metrics."""
import argparse
import io
import json
from pathlib import Path

import lmdb
import numpy as np


def metrics(folder,txn):
    summary=json.loads((folder/'summary.json').read_text())
    first=[];other=[];all_errors=[];width=[]
    for r in summary['results']:
        sid=r['sample_id'];row=json.loads(txn.get(sid.encode()))
        with np.load(io.BytesIO(txn.get(row['trajectory_key'].encode())),allow_pickle=False) as z:
            target=z['points'];starts=z['stroke_offsets'][:-1]
        with np.load(folder/sid/'teacher_forced.npz',allow_pickle=False) as z:pred=z['geometry']
        error=np.linalg.norm(pred[:,:2]-target[:,:2],axis=-1)
        first.extend(error[starts].tolist());keep=np.ones(len(error),bool);keep[starts]=False
        other.extend(error[keep].tolist());all_errors.extend(error.tolist());width.extend(np.abs(pred[:,2]-target[:,2]).tolist())
    return dict(samples=summary['samples'],teacher_forced_iou=summary['mean_teacher_forced_iou'],
                image_only_iou=summary['mean_image_only_iou'],completed=summary['completed'],
                exact_stroke_count=summary['exact_stroke_count'],start_l2_px=float(np.mean(first)),
                other_l2_px=float(np.mean(other)),xy_l2_px=float(np.mean(all_errors)),
                points_within_2px=float(np.mean(np.asarray(all_errors)<=2)),
                points_within_5px=float(np.mean(np.asarray(all_errors)<=5)),width_mae_px=float(np.mean(width)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();base=Path(args.baseline);candidate=Path(args.candidate)
    a=json.loads((base/'summary.json').read_text());b=json.loads((candidate/'summary.json').read_text())
    if [r['sample_id'] for r in a['results']]!=[r['sample_id'] for r in b['results']]:
        raise ValueError('Comparison requires the same ordered evaluation image IDs')
    with lmdb.open(args.dataset,readonly=True,lock=False,readahead=False) as env:
        with env.begin() as txn:
            old=metrics(base,txn);new=metrics(candidate,txn)
    result=dict(baseline=str(base),candidate=str(candidate),baseline_metrics=old,candidate_metrics=new,
                candidate_minus_baseline={k:new[k]-old[k] for k in old if k!='samples'},
                note='Lower pixel error is better; higher IoU/within-threshold fraction is better. Weighted losses have different units and are not compared.')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
