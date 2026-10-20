"""Export only explicitly approved region masks into white-background VS inputs."""
import argparse
from common import records, unique, output_dir, arrays, masks_and_labels, require, png, write_records, digest


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); rows=records(a.manifest); unique(rows); out=output_dir(a.output); result=[]
    for r in rows:
        require(r['review_status']=='approved' and r.get('mask_review'),'Mask review approval required: '+r['sample_id'])
        require(r['reviewed_sha256']==digest(r['reviewed_path']),'Reviewed masks changed')
        z=arrays(r['reviewed_path']); masks=z['extractnet_strokes']; labels=z['stroke_labels']; masks_and_labels(masks,labels)
        for i,m in enumerate(masks):
            sid=r['sample_id']+'_stroke%03d'%i
            path=png(out/(sid+'.png'),(1-m)*255)
            result.append(dict(r,stroke_id=sid,stroke_index=i,stroke_label=int(labels[i]),expected_strokes=len(masks),input_png=path,input_sha256=digest(path)))
    write_records(out/'manifest.jsonl',result)


if __name__=='__main__': main()
