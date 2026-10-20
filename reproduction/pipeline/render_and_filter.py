"""Decode/rerender candidates, measure source IoU and independent upstream fidelity."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import records, output_dir, arrays, gray, png, iou, render, validate_points, read_json, write_records, write_json, preview, digest, require, identifier
from decode_vs_npz import decode, replay_native


def evaluate(z, mask, upstream, minimum=0.95, fidelity=0.95):
    rendered=render(z['points'],z['subpath_offsets'])
    score=iou(rendered>=128,mask)
    consistency=iou(rendered>=128,upstream<128)
    reasons=[]
    try: validate_points(z['points'],z['subpath_offsets'])
    except ValueError as e: reasons.append(str(e))
    if not np.any(mask): reasons.append('empty target')
    if score<minimum: reasons.append('source IoU below threshold')
    if consistency<fidelity: reasons.append('upstream rendering consistency below threshold')
    return rendered,dict(iou=score,upstream_iou=consistency,qc_status='passed' if not reasons else 'rejected',reasons=reasons,threshold=minimum,fidelity_threshold=fidelity,renderer='continuous round brush, supersample=4, binary >=128')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--iou',type=float,default=.95); p.add_argument('--fidelity',type=float,default=.95); p.add_argument('--spacing',type=float,default=.5)
    a=p.parse_args(); require(0<a.iou<=1 and 0<a.fidelity<=1,'Invalid thresholds'); out=output_dir(a.output); results=[]; errors=[]; groups=defaultdict(list)
    rows=records(a.manifest); require(len({r['candidate_id'] for r in rows})==len(rows),'Duplicate candidate ID')
    for r in rows:
        try:
            cid=identifier(r['candidate_id']); d=out/cid; d.mkdir()
            for path_key,hash_key in [('npz','npz_sha256'),('input_png','input_sha256'),('upstream_render_path','upstream_render_sha256')]:
                if r.get(hash_key): require(digest(r[path_key])==r[hash_key],'Candidate input changed: '+path_key)
            z=arrays(r['trajectory_path']) if r.get('trajectory_path') else decode(r['npz'],read_json(r['model_config']),a.spacing)
            mask=gray(r['input_png'])<128; upstream=gray(r['upstream_render_path'])
            image,qc=evaluate(z,mask,upstream,a.iou,a.fidelity)
            replay=replay_native(r['npz'],read_json(r['model_config']))
            qc['native_replay_iou']=iou(replay>=128,upstream<128)
            if qc['native_replay_iou']<.999:
                qc['qc_status']='rejected'; qc['reasons'].append('Independent native replay mismatch')
            png(d/'native_replay.png',255-replay)
            path=d/'trajectory.npz'; np.savez_compressed(str(path),**z)
            overlay=np.full((256,256,3),255,np.uint8); overlay[mask]=[0,0,255]; overlay[image>=128]=[255,0,0]; overlay[mask & (image>=128)]=[0,0,0]
            item=dict(r,trajectory_path=str(path),trajectory_sha256=digest(path),render_path=png(d/'render.png',255-image),overlay_path=png(d/'overlay.png',overlay),qc=qc,trajectory_review_status='pending')
            results.append(item); groups[r['stroke_id']].append(item)
        except Exception as e: errors.append(dict(candidate_id=r.get('candidate_id'),error=str(e)))
    best=[]
    for sid,items in groups.items():
        passing=[r for r in items if r['qc']['qc_status']=='passed']
        best.append(max(passing or items,key=lambda r:r['qc']['iou']))
    write_records(out/'manifest.jsonl',results); write_records(out/'best.jsonl',best); write_records(out/'errors.jsonl',errors)
    write_json(out/'summary.json',dict(candidates=len(results),passed=sum(r['qc']['qc_status']=='passed' for r in results),errors=len(errors)))
    preview(out,best,'Best candidate per stroke: red=target, blue=render, black=intersection')
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
