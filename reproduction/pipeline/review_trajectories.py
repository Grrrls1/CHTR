"""Review direction: reorder/reverse subpaths or replace points; rerun QC after edits."""
import argparse
from pathlib import Path
import numpy as np
from common import records, output_dir, arrays, read_json, write_json, write_records, digest, require, states, validate_points, gray, png, audit, preview, interactive_decisions, identifier, iou
from decode_vs_npz import replay_native
from render_and_filter import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True); p.add_argument('--decisions')
    p.add_argument('--interactive',help='Edit an existing decisions.template.json in the terminal')
    a=p.parse_args(); require(not(a.decisions and a.interactive),'Choose decisions or interactive')
    rows=records(a.manifest); out=output_dir(a.output); decisions=interactive_decisions(a.interactive) if a.interactive else read_json(a.decisions) if a.decisions else {}; result=[]; template={}
    require(len({r['stroke_id'] for r in rows})==len(rows),'Select one candidate per stroke before review')
    require(set(decisions)<=set(r['stroke_id'] for r in rows),'Unknown decision ID')
    for r in rows:
        identifier(r['stroke_id'])
        for path_key,hash_key in [('trajectory_path','trajectory_sha256'),('input_png','input_sha256'),('upstream_render_path','upstream_render_sha256')]:
            if r.get(hash_key): require(digest(r[path_key])==r[hash_key],'Review input changed: '+path_key)
        z=arrays(r['trajectory_path']); decision=decisions.get(r['stroke_id']); status='pending'; history=[]
        if decision and decision.get('status')=='pending': decision=None
        original=digest(r['trajectory_path'])
        if decision:
            require(decision['source_sha256']==original,'Stale trajectory review')
            require(decision['status'] in ('approved','rejected'),'Explicit approved/rejected required')
            history=list(r.get('trajectory_review',[]))+[audit(decision['reviewer'],decision,r['trajectory_path'])]
            if decision.get('replacement_path'):
                z=arrays((Path(a.decisions or a.interactive).resolve().parent/decision['replacement_path']).resolve())
            elif decision.get('subpaths') is not None:
                ops=decision['subpaths']; n=len(z['subpath_offsets'])-1
                require(sorted(o['index'] for o in ops)==list(range(n)),'Every subpath must occur once; use replacement NPZ for geometry/deletion edits')
                paths=[]
                for op in ops:
                    i=op['index']; lo,hi=z['subpath_offsets'][i:i+2]; part=z['points'][lo:hi].copy()
                    paths.append(part[::-1] if op.get('reverse',False) else part)
                offsets=np.r_[0,np.cumsum([len(v) for v in paths])].astype(np.int64)
                z=dict(points=states(np.concatenate(paths),offsets),subpath_offsets=offsets,image_size=np.array(256))
            validate_points(z['points'],z['subpath_offsets'],bounds=False)
            status=decision['status']
        image,qc=evaluate(z,gray(r['input_png'])<128,gray(r['upstream_render_path']),r['qc']['threshold'],r['qc']['fidelity_threshold'])
        if r.get('npz') and r.get('model_config'):
            replay=replay_native(r['npz'],read_json(r['model_config']))
            qc['native_replay_iou']=iou(replay>=128,gray(r['upstream_render_path'])<128)
            if qc['native_replay_iou']<.999:
                qc['qc_status']='rejected'; qc['reasons'].append('Independent native replay mismatch')
        require(status!='approved' or qc['qc_status']=='passed','Cannot approve failed QC: '+r['stroke_id'])
        d=out/r['stroke_id']; d.mkdir(); dest=d/'reviewed.npz'; np.savez_compressed(str(dest),**z)
        item=dict(r,trajectory_path=str(dest),trajectory_sha256=digest(dest),render_path=png(d/'render.png',255-image),qc=qc,trajectory_review_status=status,trajectory_review=history)
        result.append(item)
        template[r['stroke_id']]=dict(source_sha256=original,status='pending',reviewer='',subpaths=[dict(index=i,reverse=False) for i in range(len(z['subpath_offsets'])-1)])
    write_records(out/'manifest.jsonl',result); write_json(out/'decisions.template.json',template); preview(out,result,'Trajectory review: numbered starts and direction arrows')


if __name__=='__main__': main()
