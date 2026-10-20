"""Reassemble every cleanup-qualified sample; IoU is metadata, not an admission gate."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import tempfile
import time

from build_chtr_dataset import ROOT, PIPELINE, LABEL_NAMES, finalize_manifest
from continue_chtr_dataset import checked_output, cleanup_summary, exclusive_run
from continue_skeleton_dataset import TYPE_MAP, validate_training_row
sys.path.insert(0,str(PIPELINE));sys.path.insert(0,str(ROOT/'CHTR'))
from common import digest, iter_records, read_json, records, require, write_json, write_records
from assemble_single_path import assemble_group
from models.datasets.lmdb_dataset import load_type_map


def assemble_one(group, output, threshold):
    try:
        return assemble_group(group, Path(output), threshold), None
    except ValueError as exc:
        return None, dict(sample_id=group[0]['sample_id'],error=str(exc))


def rebuild_batch(batch, output, names, config, pool):
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'dataset.done.json'
    completed=checked_output(marker)
    if completed is not None:
        return completed/'manifest.jsonl'
    export=checked_output(batch/'export.done.json')
    require(export is not None,'Missing completed export: '+str(batch))
    exports=records(export/'manifest.jsonl')
    by_id=defaultdict(list)
    for row in exports:by_id[row['sample_id']].append(row)
    sources=[row for row in iter_records(batch/'cleanup.jsonl') if row['sample_id'] in by_id]
    raw=checked_output(batch/'skeleton.done.json') if exports else None
    all_strokes=records(raw/'manifest.jsonl') if raw is not None else []
    raw_errors=records(raw/'errors.jsonl') if raw is not None else []
    groups=defaultdict(list)
    for row in all_strokes:groups[row['sample_id']].append(row)
    chars=Path(tempfile.mkdtemp(prefix='characters-',dir=str(output)))
    packed=[];rejected=[]
    jobs=[pool.submit(assemble_one,group,str(chars),config['quality_threshold']) for group in groups.values()]
    for job in jobs:
        item,error=job.result()
        if item is not None:packed.append(item)
        if error is not None:rejected.append(error)
    write_records(chars/'manifest.jsonl',packed);write_records(chars/'rejected.jsonl',rejected)
    dataset=Path(tempfile.mkdtemp(prefix='dataset-',dir=str(output)))
    stroke_manifest=output/'all_strokes.jsonl';write_records(stroke_manifest,all_strokes)
    # Small per-batch report from the already completed run, never the global report.
    finalize_manifest(sources,stroke_manifest,chars/'manifest.jsonl',batch/'cleanup_report.json',
                      dataset,config['quality_threshold'],config['trajectory_style'],False)
    output_rows=[];invalid=[]
    error_by_id={r['stroke_id']:r['error'] for r in raw_errors}
    reports={r['sample_id']:r for r in read_json(batch/'cleanup_report.json')['samples']}
    for row in iter_records(dataset/'manifest.jsonl'):
        sid=row['sample_id']
        if row['build_status']=='trajectory_ready':
            try:validate_training_row(row,names,1024)
            except ValueError as exc:
                row['rejected_trajectory_path']=row['trajectory_path'];row['trajectory_path']=None
                row['build_status']='trajectory_pending';row['pending_reason']='model_contract_rejected'
                row['quality']['passed']=False;invalid.append(dict(sample_id=sid,error=str(exc)))
        if row['build_status']=='trajectory_pending':
            strokes=sorted(by_id[sid],key=lambda r:r['stroke_index'])
            row['stroke_count']=len(strokes)
            row['stroke_types']=[int(r['stroke_label']) for r in strokes]
            row['stroke_type_names']=[LABEL_NAMES[label] for label in row['stroke_types']]
            row['stroke_map']=[dict(stroke_index=r['stroke_index'],stroke_id=r['stroke_id'],
                stroke_label=int(r['stroke_label']),stroke_type_name=LABEL_NAMES[int(r['stroke_label'])],
                mask_path=r['input_png'],point_range=None) for r in strokes]
            row['pending_reason']=row.get('pending_reason') if row.get('pending_reason')=='model_contract_rejected' else 'incomplete_skeleton_trajectory'
            row['trajectory_errors']=[dict(stroke_id=r['stroke_id'],error=error_by_id[r['stroke_id']])
                                      for r in strokes if r['stroke_id'] in error_by_id]
            report=reports[sid]
            row['stage_iou']=dict(mask_extraction_whole=report['before']['iou'],mask_cleanup_whole=report['after']['iou'])
        row['dataset_inclusion']='cleanup_iou_ge_0.87'
        row['training_eligible']=row['build_status']=='trajectory_ready'
        output_rows.append(row)
    write_records(dataset/'manifest.jsonl',output_rows)
    write_records(dataset/'model_rejected.jsonl',invalid)
    count=Counter(r['build_status'] for r in output_rows)
    write_json(dataset/'summary.json',dict(samples=len(output_rows),ready=count['trajectory_ready'],
        pending=count['trajectory_pending'],quality_passed=sum(r['quality']['passed'] for r in output_rows),
        model_rejected=len(invalid),raw_errors=len(raw_errors)))
    write_json(marker,dict(output=str(dataset),hashes={name:digest(dataset/name) for name in
        ('manifest.jsonl','summary.json','model_rejected.jsonl')}))
    print('[batch] '+json.dumps(read_json(dataset/'summary.json')),flush=True)
    return dataset/'manifest.jsonl'


def merge_batches(manifests,output,expected):
    counts=Counter()
    def rows():
        for manifest in manifests:
            for row in iter_records(manifest):
                counts['samples']+=1;counts[row['build_status']]+=1
                counts['quality_passed']+=bool(row['quality'].get('passed'))
                yield row
    write_records(output/'manifest.jsonl',rows())
    require(counts['samples']==expected,'Candidate count mismatch: %s vs %s'%(counts['samples'],expected))
    write_records(output/'training_manifest.jsonl',
                  (r for r in iter_records(output/'manifest.jsonl') if r['training_eligible']))
    summary=dict(samples=counts['samples'],ready=counts['trajectory_ready'],pending=counts['trajectory_pending'],
                 quality_passed=counts['quality_passed'],inclusion='cleanup IoU >= 0.87; trajectory IoU is metadata only')
    write_json(output/'summary.json',summary)
    return summary


def pack_all(manifest,destination,type_map,summary,map_gb=64):
    import lmdb
    from pack_chtr_lmdb import pack_masks
    signature=dict(manifest_sha256=digest(manifest),type_map_sha256=digest(type_map))
    if (destination/'complete.json').exists():
        require(read_json(destination/'complete.json')['signature']==signature,'Rebuild LMDB signature changed')
        return
    require(not destination.exists(),'Incomplete build directory; use a new --lmdb-build')
    destination.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.all_clean_lmdb.partial-',dir=str(destination.parent)))
    env=lmdb.open(str(work),map_size=int(map_gb*1024**3))
    ids=[];ready=[];seen=set();txn=env.begin(write=True)
    try:
        for row in iter_records(manifest):
            sid=row['sample_id'];require(sid not in seen,'Duplicate ID: '+sid);seen.add(sid);ids.append(sid)
            meta=dict(row,lmdb_key=sid,image_key=sid+'/image',masks_key=sid+'/masks',
                      original_image_key=sid+'/original' if row.get('original_image_path') else None,
                      trajectory_key=sid+'/trajectory' if row['training_eligible'] else None)
            for key,field in (('image_key','image_path'),('original_image_key','original_image_path'),('trajectory_key','trajectory_path')):
                if meta[key]:txn.put(meta[key].encode(),Path(row[field]).read_bytes())
            masks=pack_masks(row,manifest.parent);require(masks is not None,'Missing masks: '+sid)
            txn.put(meta['masks_key'].encode(),masks)
            txn.put(sid.encode(),json.dumps(meta,ensure_ascii=False,allow_nan=False).encode())
            if meta['trajectory_key']:ready.append(sid)
            if len(ids)%128==0:
                txn.commit();txn=env.begin(write=True);print('[lmdb] packed=%d ready=%d'%(len(ids),len(ready)),flush=True)
        require(len(ids)==summary['samples'],'LMDB count mismatch')
        metadata=dict(summary,schema='chtr-lmdb-v1',count=len(ids),include_masks=True,include_original=True,
                      manifest=str(manifest),training_count=len(ready))
        txn.put(b'__ids__',json.dumps(ids).encode());txn.put(b'__ids_ready__',json.dumps(ready).encode())
        txn.put(b'__meta__',json.dumps(metadata).encode());txn.commit();txn=None;env.sync()
    finally:
        if txn is not None:txn.abort()
        env.close()
    shutil.copyfile(type_map,work/'stroke_type_map.json')
    write_json(work/'summary.json',metadata);write_json(work/'complete.json',dict(signature=signature))
    work.rename(destination)


def publish(build,destination):
    # chtr_train.slurm holds a shared lock for the duration of training.
    with (destination.parent/'.chtr_mmah_lmdb.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        backup=None
        if destination.exists():
            require(destination.is_dir() and not destination.is_symlink(),'Unexpected dataset destination')
            backup=destination.with_name(destination.name+'.backup_'+time.strftime('%Y%m%d_%H%M%S'))
            require(not backup.exists(),'Backup path exists');destination.rename(backup)
        try:build.rename(destination)
        except Exception:
            if backup is not None:backup.rename(destination)
            raise
        print('[published] '+str(destination)+' backup='+str(backup),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output')
    p.add_argument('--dataset',default=str(ROOT/'CHTR/data/chtr_mmah_lmdb'))
    p.add_argument('--workers',type=int,default=3);p.add_argument('--map-size-gb',type=float,default=64)
    p.add_argument('--stroke-type-map',default=str(TYPE_MAP));p.add_argument('--publish',action='store_true')
    p.add_argument('--max-batches',type=int);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();require(1<=args.workers<=16,'Invalid worker count')
    run=Path(args.run).resolve();old=run/'skeleton_continuation_v1'
    output=Path(args.output).resolve() if args.output else run/'skeleton_all_clean_v1'
    require(output!=old and output!=run and output not in run.parents,'Use a separate rebuild output')
    config=read_json(old/'config.json')['config'];_,names=load_type_map(args.stroke_type_map)
    expected=cleanup_summary(run/'mask_cleanup/report.json')['status_counts']['clean_candidate']
    batches=sorted((old/'batches').glob('batch_*'))
    require(batches,'No completed skeleton batches')
    if args.dry_run:
        first=checked_output(batches[0]/'skeleton.done.json');require(first is not None,'First skeleton batch missing')
        print('Reuse %d existing batches; include %d cleanup-qualified samples; no IoU exclusion.'%(len(batches),expected))
        print('Output='+str(output)+' dataset='+args.dataset);return
    assets=[Path(__file__).resolve(),ROOT/'CHTR/data/build_chtr_dataset.py',ROOT/'CHTR/models/datasets/lmdb_dataset.py',
            ROOT/'CHTR/data/continue_skeleton_dataset.py',PIPELINE/'assemble_single_path.py',PIPELINE/'common.py']
    signature=dict(source_config=digest(old/'config.json'),type_map=digest(args.stroke_type_map),
                   assets={str(f):digest(f) for f in assets},policy='all_cleanup_candidates')
    with exclusive_run(output):
        state=output/'config.json'
        if state.exists():require(read_json(state)==signature,'Rebuild inputs/code changed; use a new output')
        else:write_json(state,signature)
        manifests=[];folder=output/'batches';folder.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for index,batch in enumerate(batches):
                if args.max_batches is not None and index>=args.max_batches:
                    print('[limited run] final merge deferred',flush=True);return
                manifests.append(rebuild_batch(batch,folder/batch.name,names,config,pool))
                print('[progress] batches=%d/%d'%(index+1,len(batches)),flush=True)
        summary=merge_batches(manifests,output,expected)
        shutil.copyfile(args.stroke_type_map,output/'stroke_type_map.json')
        destination=Path(args.dataset).resolve();build=destination.with_name('.'+destination.name+'.all_clean_build')
        # Already published after a prior interrupted final log: do not overwrite the backup.
        if (destination/'complete.json').exists() and read_json(destination/'complete.json').get('signature',{}).get('manifest_sha256')==digest(output/'manifest.jsonl'):
            print('[done] already published',flush=True);return
        pack_all(output/'manifest.jsonl',build,args.stroke_type_map,summary,args.map_size_gb)
        if args.publish:publish(build,destination)
        print('[done] '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
