"""Validate packed geometry, states, review provenance, rendering and split leakage."""
import argparse
from collections import defaultdict
import numpy as np
from common import records, arrays, unique, require, validate_points, digest, gray, iou, render, write_json, states


def validate(rows):
    errors=[]; groups=defaultdict(set); images=defaultdict(set)
    try: unique(rows); require(rows,'Dataset is empty')
    except ValueError as e: errors.append(str(e))
    for r in rows:
        try:
            require(r['split'] in ('train','test'),'Invalid split')
            groups[r['split_group']].add(r['split']); images[r['image_sha256']].add(r['split'])
            require(digest(r['npz_path'])==r['npz_sha256'] and digest(r['image_path'])==r['image_sha256'],'Packed artifact changed')
            z=arrays(r['npz_path']); p=z['points']; offsets=z['subpath_offsets']; validate_points(p,offsets)
            require(p.dtype==np.float32,'points must be float32')
            so=z['stroke_offsets']; labels=z['stroke_types']; n=r['expected_strokes']
            require(np.issubdtype(so.dtype,np.integer) and so.shape==(n+1,) and so[0]==0 and so[-1]==len(p) and np.all(np.diff(so)>0),'Bad stroke offsets')
            require(set(so)<=set(offsets),'Stroke boundary must be a subpath boundary')
            require(np.issubdtype(labels.dtype,np.integer) and labels.shape==(n,) and np.all((labels>=0)&(labels<24)),'Bad category IDs')
            require(len(r['strokes'])==n,'Incomplete review provenance')
            for i,stroke in enumerate(r['strokes']):
                require(stroke.get('mask_review') and stroke.get('trajectory_review') and stroke['review_status']=='approved' and stroke['trajectory_review_status']=='approved','Missing approval')
                require(stroke['qc']['qc_status']=='passed' and stroke['qc']['iou']>=.95 and stroke['qc']['upstream_iou']>=.95,'Unqualified stroke')
                require(stroke['stroke_index']==i and stroke['stroke_label']==int(labels[i]),'Review stroke order/category mismatch')
                require(digest(stroke['input_png'])==stroke['input_sha256'],'Archived stroke mask changed')
                lo,hi=so[i:i+2]; local=offsets[(offsets>=lo)&(offsets<=hi)]-lo
                part=states(p[lo:hi],local)
                require(iou(render(part,local)>=128,gray(stroke['input_png'])<128)>=max(.95,r['threshold']),'Packed stroke IoU failed')
                require(digest(stroke['upstream_render_path'])==stroke['upstream_render_sha256'],'Archived upstream render changed')
                require(iou(render(part,local)>=128,gray(stroke['upstream_render_path'])<128)>=max(.95,r['threshold']),'Packed stroke fidelity failed')
            score=iou(render(p,offsets)>=128,gray(r['image_path'])<128)
            require(score>=max(.95,r['threshold']),'Whole-character rendering failed')
            require(abs(score-r['whole_iou'])<1e-8,'Stored IoU mismatch')
        except (KeyError,ValueError,OSError) as e: errors.append(r.get('sample_id','?')+': '+str(e))
    for kind,mapping in [('writer/style',groups),('identical image',images)]:
        for key,splits in mapping.items():
            if len(splits)>1: errors.append(kind+' leakage: '+key)
    return dict(valid=not errors,samples=len(rows),errors=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--report',required=True)
    a=p.parse_args(); report=validate(records(a.manifest)); write_json(a.report,report); print(report)
    if not report['valid']: raise SystemExit(1)


if __name__=='__main__': main()
