"""Generate review previews; apply explicit versioned mask edits and decisions."""
import argparse
from pathlib import Path
import numpy as np
from common import records, unique, output_dir, arrays, masks_and_labels, png, preview, write_records, read_json, write_json, gray, audit, require, digest, interactive_decisions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--decisions',help='JSON: sample_id -> {source_sha256,status,reviewer,strokes:[{indices:[0],label:5,mask_path:optional}]}')
    p.add_argument('--interactive',help='Interactively edit a previously generated decisions.template.json')
    a=p.parse_args(); rows=records(a.manifest); unique(rows); out=output_dir(a.output)
    require(not(a.decisions and a.interactive),'Choose decisions or interactive')
    decisions=interactive_decisions(a.interactive) if a.interactive else read_json(a.decisions) if a.decisions else {}; template={}; result=[]; views=[]
    require(set(decisions)<=set(r['sample_id'] for r in rows),'Unknown decision ID')
    for r in rows:
        source=r.get('reviewed_path',r['extracted_path']); z=arrays(source); masks=z['extractnet_strokes']; labels=z['stroke_labels']; masks_and_labels(masks,labels)
        decision=decisions.get(r['sample_id']); status='pending'; history=[]
        if decision and decision.get('status')=='pending': decision=None
        if decision:
            require(decision['source_sha256']==digest(source),'Stale mask review')
            require(decision['status'] in ('approved','rejected'),'Explicit approved/rejected status required')
            history=list(r.get('mask_review',[]))+[audit(decision['reviewer'],decision,source)]
            if 'strokes' in decision:
                require(isinstance(decision['strokes'],list) and decision['strokes'],'Edited stroke list must be nonempty')
                edited=[]; labs=[]
                for op in decision['strokes']:
                    require(type(op['label']) is int and 0<=op['label']<24,'Edited label must be an integer 0..23')
                    if op.get('mask_path'):
                        path=(Path(a.decisions or a.interactive).resolve().parent/op['mask_path']).resolve(); m=(gray(path)<128).astype(np.uint8)
                    else:
                        ids=op['indices']; require(ids and all(isinstance(i,int) and 0<=i<len(masks) for i in ids),'Bad stroke indices')
                        m=np.any(masks[ids],axis=0).astype(np.uint8)
                    edited.append(m); labs.append(op['label'])
                masks=np.asarray(edited); labels=np.asarray(labs,dtype=np.int64)
                masks_and_labels(masks,labels)
            status=decision['status']
        dest=out/r['sample_id']; dest.mkdir()
        npz=dest/'reviewed.npz'; np.savez_compressed(str(npz),extractnet_strokes=masks,stroke_labels=labels)
        item=dict(r,reviewed_path=str(npz),review_status=status,expected_strokes=len(masks),mask_review=history,reviewed_sha256=digest(npz))
        result.append(item)
        for i,m in enumerate(masks): views.append(dict(sample_id=r['sample_id'],stroke_index=i,stroke_label=int(labels[i]),review_status=status,preview_path=png(dest/('stroke%03d.png'%i),(1-m)*255)))
        template[r['sample_id']]=dict(source_sha256=digest(source),status='pending',reviewer='',strokes=[dict(indices=[i],label=int(v)) for i,v in enumerate(z['stroke_labels'])])
    write_records(out/'manifest.jsonl',result); write_json(out/'decisions.template.json',template); preview(out,views,'Stroke masks: review order, region and category')


if __name__=='__main__': main()
