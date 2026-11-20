"""Frozen, image-only HiWTR cross-domain test-subset repeated sampling.

No training configuration or continuation guards are changed by this adapter.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.datasets.casia_training import sha256_file
from models.training.single_stroke import make_catalog, load_model as load_point
from models.training.ordered_stroke import load_model as load_stroke
from models.networks.two_stage import decode_stroke_instances
from models.training.two_stage_metrics import compare_trajectories, summarize
from models.training.casia_benchmark_metrics import trajectory_metrics, PROTOCOL
from models.training.single_stroke_evaluation import iou

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
 'M_CASIA': ('models/runs/casia_to_calligraphy/frozen_weights/stroke_casia_job1597_epoch006_step151632.pt',
             'models/runs/casia_olhwdb11/frozen_points/job1136_epoch004_step1120224.pt', 32, 256),
 'M_BOOK': ('models/runs/ordered_stroke_full/job_679/train/ordered_stroke.pt',
            'models/runs/single_stroke_full/job_605/train/single_stroke.pt', 64, 400),
 'M_TRANSFER': ('models/runs/casia_to_calligraphy/ordered_stroke/job_3765/train/best.pt',
                'models/runs/casia_to_calligraphy/frozen_weights/point_job3215_epoch013_step29276.pt', 64, 400)}
DOMAINS = {
 'CASIA': dict(dataset='data/CASIA-OLHWDB', split_file='data/CASIA-OLHWDB/training_v1/split.json', dataset_backend='casia_pot'),
 'BOOK': dict(dataset='data/chtr_mmah_lmdb', split_file='models/data_splits/full_character_v1.json', dataset_backend='calligraphy_lmdb')}
EXPECTED = {
 'CASIA': ('3b4a8687c3a250559d03f690ce91f91cb7727be91f7dadc0748f99695bdd7701','efd52a789fad7a550cca752d535c46e5be1b638ab72929c9bd5753821d339b1a',224559),
 'BOOK': ('124790ac1671c8892db829c7d611c35d80171142e81b46fe469a48f6535f3563','337d969a0a44da80e3aca8c52f0fa6b1fc0f6f5fbdc0a005e2877e1fe02925f8',6714)}
EXPECTED_WEIGHTS = {
 MODELS['M_CASIA'][0]:'c4402823a5eeade177bbad478b24c97555f15934b1deb1d15324d6b99d8fc721',
 MODELS['M_CASIA'][1]:'e0cd0b92536b4213073f1886f9553534244bba0d7e83be14d3baad34bbae7043',
 MODELS['M_TRANSFER'][1]:'a95ae7b889a46f185f3de795be87059e8352665ef4ddfd8315776529592633d7'}
SEEDS = list(range(20261120,20261125))
VERSION = 'hiwtr_sampled_cross_domain_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    s = io.StringIO(); keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(s, fieldnames=keys); w.writeheader(); w.writerows(rows)
    atomic_bytes(path,s.getvalue().encode())


def sample_manifests(ids, n, seeds=SEEDS):
    ids = sorted(ids)
    if len(set(ids)) != len(ids) or not 0 < n <= len(ids):
        raise ValueError('unique nonempty candidate IDs and valid sample count required')
    return [dict(seed=seed, ids=np.random.default_rng(seed).choice(ids,n,replace=False).tolist(),
                 candidate_count=len(ids), candidate_ids_sha256=digest(ids), numpy_version=np.__version__,
                 sampling_algorithm='sorted IDs; numpy.random.default_rng(seed).choice(replace=False)',
                 split='test', full_split=False, evaluation_scope='test_subset') for seed in seeds]


def source_revision():
    paths = sorted(set(ROOT.glob('models/**/*.py')))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                working_tree=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in paths})


def prepare(out):
    if (out/'config.json').exists():
        raise ValueError('already frozen; use smoke/run to resume')
    out.mkdir(parents=True,exist_ok=True)
    train = ROOT/'models/runs/casia_to_calligraphy/ordered_stroke/job_3765/train'
    last = torch.load(train/'last.pt',map_location='cpu',weights_only=False)
    cutoff = last['last_evaluated_epoch']
    summary = read_json(train/f'eval_epoch_{cutoff:03d}/summary.json')
    assert summary['full_split'] is True and summary['characters']==8121
    metrics = [json.loads(line) for line in (train/'metrics.jsonl').read_text().splitlines()]
    # The training journal must contain the committed full validation summary.
    assert any(r.get('kind')=='validation' and r.get('epoch')==cutoff and r.get('full_split') is True for r in metrics), {r.get('kind') for r in metrics}
    proof = dict(cutoff_completed_validation_epoch=cutoff, last_checkpoint_epoch=last['epoch'],
                 last_checkpoint_step=last['step'], last_evaluated_epoch=cutoff,
                 best_selection_score=last['best_selection_score'], selection_metric=last['selection_metric'],
                 summary=summary, frozen_at_utc=datetime.now(timezone.utc).isoformat())
    del last
    weights={}
    for mid,(sp,pp,sc,pc) in MODELS.items():
        refs=[]
        for stage,src,capkey,cap in [('stroke',sp,'max_strokes',sc),('point',pp,'max_points',pc)]:
            dst=out/'frozen_weights'/f'{mid}_{stage}.pt'; dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/src,dst)
            h=sha256_file(dst)
            assert h==sha256_file(ROOT/src), 'source changed during freeze: '+src
            if src in EXPECTED_WEIGHTS: assert h==EXPECTED_WEIGHTS[src],src
            ck=torch.load(dst,map_location='cpu',weights_only=False)
            assert ck['config']['model'][capkey]==cap
            if mid=='M_TRANSFER' and stage=='stroke':
                assert ck['epoch']<=cutoff and ck['last_evaluated_epoch']<=cutoff
                assert ck['best_selection_score']==proof['best_selection_score']
                assert ck['selection_metric']==proof['selection_metric']
            refs.append(dict(path=str(dst.resolve()),sha256=h,original_path=src,epoch=ck['epoch'],step=ck['step'],
                             training_mode=ck['training_mode'],signature=ck.get('signature'),
                             selection_metric=ck.get('selection_metric','historical val ordered-mask IoU' if stage=='stroke' else 'historical point validation'),
                             selection_score=ck.get('best_selection_score',ck.get('best_val_ordered_iou')),
                             training_config=ck['config']))
            dst.chmod(0o444); del ck
        weights[mid]=dict(stroke=refs[0],point=refs[1],max_strokes=sc,max_points=pc)
    domains={}
    for domain,cfg in DOMAINS.items():
        catalog=make_catalog(cfg)
        try:
            fp,sh,count=EXPECTED[domain]
            assert catalog.fingerprint==fp and sha256_file(ROOT/cfg['split_file'])==sh
            assert len(catalog.split['test_ids'])==count
            manifests=sample_manifests(catalog.split['test_ids'],100)
            for m in manifests:
                m.update(dataset_fingerprint=fp,split_sha256=sh)
                p=out/'manifests'/domain/f"seed_{m['seed']}.json"; write_json(p,m); p.chmod(0o444)
            union=sorted(set(s for m in manifests for s in m['ids']))
            render=getattr(catalog,'protocol',dict(size=256,source='stored grayscale and original XYW',synthetic_width=False))
            domains[domain]=dict(config=cfg,fingerprint=fp,split_sha256=sh,render=render,
                candidate_count=count,candidate_ids_sha256=digest(sorted(catalog.split['test_ids'])),
                union_ids=union,unique_ids=len(union),repeat_overlap_count=500-len(union),
                pairwise_overlap=[[len(set(a['ids'])&set(b['ids'])) for b in manifests] for a in manifests],
                smoke_val_ids=sorted(catalog.split['val_ids'])[:3],
                manifest_sha256={str(m['seed']):sha256_file(out/'manifests'/domain/f"seed_{m['seed']}.json") for m in manifests})
        finally: catalog.close()
    config=dict(version=VERSION,split='test',samples_per_repeat_per_dataset=100,repeats=5,sampling_seeds=SEEDS,
        sampling_unit='character_image_sample',sampling_method='uniform_without_replacement_within_repeat',
        protocol='native_caps_v1',precision='FP32',character_batch_size=1,point_batch_size=16,
        validity_threshold=.5,mask_threshold=.5,weights=weights,domains=domains,
        full_split=False,evaluation_scope='test_subset',test_used_for_model_selection=False,
        metric_protocol=PROTOCOL,freeze_proof=proof,sd_ddof=1)
    write_json(out/'code_revision.json',source_revision())
    write_json(out/'config.json',config); (out/'config.json').chmod(0o444)
    write_json(out/'status.json',dict(state='frozen',completed={}))
    (out/'failures.jsonl').touch()
    print('Frozen:', {d:v['unique_ids'] for d,v in domains.items()},flush=True)


def validate_frozen(out):
    cfg=read_json(out/'config.json')
    revision=read_json(out/'code_revision.json')
    assert source_revision()['source_sha256']==revision['source_sha256'], 'source changed after freeze'
    for model in cfg['weights'].values():
        for stage in ('stroke','point'):
            assert sha256_file(model[stage]['path'])==model[stage]['sha256']
    for domain,d in cfg['domains'].items():
        assert sha256_file(ROOT/d['config']['split_file'])==d['split_sha256']
        for seed,h in d['manifest_sha256'].items():
            assert sha256_file(out/'manifests'/domain/f'seed_{seed}.json')==h
    return cfg


def identity(cfg,mid,domain):
    return digest(dict(model=cfg['weights'][mid],domain=cfg['domains'][domain],version=cfg['version'],
                       protocol=cfg['protocol'],precision=cfg['precision'],threshold=.5,point_batch_size=16,
                       source=read_json(Path(cfg['_output'])/'code_revision.json')['source_sha256']))


def valid_cache(record, cache_id, sid):
    return record.get('identity')==cache_id and record.get('sample_id')==sid and record.get('committed') is True


def cached_record(path, cache_id, sid):
    try:
        record=read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(record,dict) or not valid_cache(record,cache_id,sid):
        return None
    if not isinstance(record.get('metrics'),dict) or not isinstance(record.get('prediction'),dict):
        return None
    return record


def predict(stroke,point,image,sc,pc,device):
    # Only grayscale is an input to either stage; no label-derived decode lengths.
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter()
    images=torch.from_numpy(image.astype(np.float32)[None,None]/255.).to(device)
    with torch.inference_mode():
        instances=stroke.generate(images,max_strokes=sc,threshold=.5)
        result=decode_stroke_instances(point,images,instances,max_points=pc,point_threshold=.5,point_batch_size=16)[0]
    torch.cuda.synchronize(); elapsed=time.perf_counter()-began
    peaks=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result,elapsed,peaks


def score(image,points,offsets,masks,result):
    gt=[points[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    row,paths=compare_trajectories(gt,result['strokes'])
    row.update(trajectory_metrics(image,gt,paths))
    inst=result['instances']; pm=(inst['masks']>=.5).detach().cpu().numpy()
    n,m=len(gt),len(paths); common=min(n,m)
    aligned=[iou(masks[k],pm[k]) for k in range(common)]
    row.update(true_strokes=n,stroke_count_mae=abs(m-n),count_exact=float(n==m),
               strict_sequence_mask50=float(n==m and all(v>=.5 for v in aligned)),
               ordered_mask_iou=sum(aligned)/max(n,m,1),empty_masks=int(sum(not a.any() for a in pm)),
               empty_output=int(m==0),character_truncated=int(result['truncated']),
               character_terminated=int(result['terminated']),unpaired_strokes=abs(n-m))
    pred=dict(points_xyw_px=np.concatenate(paths).tolist() if paths else [],
              stroke_offsets=np.r_[0,np.cumsum([len(p) for p in paths])].tolist(),
              stop_reason=inst['stop_reason'],terminated=bool(result['terminated']),truncated=bool(result['truncated']),
              point_states=[{k:v for k,v in p.items() if k!='points'} for p in result['strokes']],
              predicted_starts_xyw_px=(inst['starts'].detach().cpu().numpy()*[255,255,256]).tolist(),
              predicted_boxes_xyxy_px=(inst['boxes'].detach().cpu().numpy()*255).tolist(),
              predicted_confidences=inst['confidences'].detach().cpu().tolist())
    return row,pred,paths,pm,gt


def aggregate(rows):
    summary=summarize(rows)
    result={k:v for k,v in summary.items() if isinstance(v,(int,float)) or v is None}
    for k in ('aiou','dtw_sum_px','ldtw_px','stroke_count_mae','count_exact','strict_sequence_mask50',
              'ordered_mask_iou','character_truncated','character_terminated','inference_seconds'):
        values=[r[k] for r in rows if r[k] is not None]
        result[k]=float(np.mean(values)) if values else None
    for k in ('true_strokes','empty_masks','empty_output','unpaired_strokes','gt_exceeds_stroke_cap','gt_strokes_exceeding_point_cap'):
        result[k]=sum(r[k] for r in rows)
    result['dtw_undefined_images']=sum(r['ldtw_px'] is None for r in rows)
    result['character_truncated_count']=sum(r['character_truncated'] for r in rows)
    result['character_truncation_denominator']=len(rows)
    result['point_truncated_count']=sum(r['truncated'] for r in rows)
    result['point_truncation_denominator']=sum(r['predicted_strokes'] for r in rows)
    result['peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in rows)
    result['peak_reserved_bytes']=max(r['peak_reserved_bytes'] for r in rows)
    return result


def mean_std(repeats):
    result={}
    for k in repeats[0]:
        vals=[r[k] for r in repeats if isinstance(r[k],(int,float))]
        result[k+'_mean']=float(np.mean(vals)) if vals else None
        result[k+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        result[k+'_defined_repeats']=len(vals)
    return result


def collect(out,cfg):
    all_repeats=[]; all_means=[]; by_combo={}
    for mid,model in cfg['weights'].items():
        for domain,d in cfg['domains'].items():
            cache_id=identity(cfg,mid,domain); rows={}
            for sid in d['union_ids']:
                record=read_json(out/'predictions'/mid/domain/(digest(sid)+'.json'))
                assert valid_cache(record,cache_id,sid)
                rows[sid]=record['metrics']
            per=list(rows.values()); by_combo[mid,domain]=rows
            atomic_bytes(out/'per_character'/f'{mid}_{domain}.jsonl', ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in per).encode())
            write_csv(out/'per_character'/f'{mid}_{domain}.csv',per)
            reps=[]
            for seed in cfg['sampling_seeds']:
                manifest=read_json(out/'manifests'/domain/f'seed_{seed}.json')
                result=aggregate([rows[sid] for sid in manifest['ids']]); reps.append(result)
                meta=dict(model_id=mid,test_domain=domain,seed=seed,characters=len(manifest['ids']),
                          candidate_test_count=d['candidate_count'],full_split=False,evaluation_scope='test_subset',
                          max_strokes=model['max_strokes'],max_points=model['max_points'])
                all_repeats.append(dict(**meta,**result))
                write_json(out/'summaries'/mid/domain/f'seed_{seed}.json',dict(**meta,**result))
            all_means.append(dict(model_id=mid,test_domain=domain,max_strokes=model['max_strokes'],
                                 max_points=model['max_points'],**mean_std(reps)))
    pairs=[]
    mids=list(cfg['weights'])
    for domain in cfg['domains']:
        for seed in cfg['sampling_seeds']:
            ids=read_json(out/'manifests'/domain/f'seed_{seed}.json')['ids']
            for i,a in enumerate(mids):
                for b in mids[i+1:]:
                    row=dict(test_domain=domain,seed=seed,model_a=a,model_b=b,characters=len(ids),difference='a_minus_b')
                    for key in ('aiou','character_trajectory_iou','ordered_trajectory_iou','ldtw_px','dtw_sum_px','stroke_count_mae'):
                        delta=[by_combo[a,domain][s][key]-by_combo[b,domain][s][key] for s in ids
                               if by_combo[a,domain][s][key] is not None and by_combo[b,domain][s][key] is not None]
                        row[key]=float(np.mean(delta)) if delta else None; row[key+'_paired_n']=len(delta)
                    pairs.append(row)
    write_csv(out/'per_repeat_metrics.csv',all_repeats)
    write_csv(out/'summary_mean_std.csv',all_means)
    write_csv(out/'paired_differences.csv',pairs)
    assert len(all_repeats)==30 and len(all_means)==6
    return {f'{m}/{d}':len(rows) for (m,d),rows in by_combo.items()}


def run(out, smoke=False):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'GPU evaluation must use Slurm'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cfg=validate_frozen(out); cfg['_output']=str(out.resolve())
    if not smoke:
        proof=read_json(out/'smoke'/'status.json'); assert proof['state']=='complete' and proof['config_sha256']==sha256_file(out/'config.json')
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(),device_properties=str(torch.cuda.get_device_properties(0)),
                     precision='FP32',tf32=False,character_batch_size=1,point_batch_size=16,
                     slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(out/('smoke/environment.json' if smoke else 'environment.json'),environment)
    completed={}
    for mid,model in cfg['weights'].items():
        stroke,sk=load_stroke(model['stroke']['path'],'cuda:0'); del sk
        point,pk=load_point(model['point']['path'],'cuda:0'); del pk
        stroke.eval(); point.eval()
        for domain,d in cfg['domains'].items():
            catalog=make_catalog(d['config'])
            assert catalog.fingerprint==d['fingerprint']
            try:
                # Fixed validation-only warmup before each domain. Never uses test labels.
                warm=catalog.character_assets(d['smoke_val_ids'][0])[0]
                for _ in range(2): predict(stroke,point,warm,model['max_strokes'],model['max_points'],'cuda:0')
                ids=d['smoke_val_ids'] if smoke else d['union_ids']
                cache_id=identity(cfg,mid,domain)
                for cursor,sid in enumerate(ids):
                    cache=out/('smoke/predictions' if smoke else 'predictions')/mid/domain/(digest(sid)+'.json')
                    if cached_record(cache,cache_id,sid) is not None: continue
                    try:
                        image,points,offsets,masks=catalog.character_assets(sid)
                        result,elapsed,peaks=predict(stroke,point,image,model['max_strokes'],model['max_points'],'cuda:0')
                        metrics,pred,paths,pm,gt=score(image,points,offsets,masks,result)
                        metrics.update(gt_exceeds_stroke_cap=int(len(offsets)-1>model['max_strokes']),
                                       gt_strokes_exceeding_point_cap=int(sum(np.diff(offsets)>model['max_points'])),
                                       sample_id=sid,character=catalog.rows[sid]['character'],model_id=mid,test_domain=domain,
                                       inference_seconds=elapsed,**peaks)
                        record=dict(identity=cache_id,committed=True,sample_id=sid,metrics=metrics,prediction=pred)
                        if not smoke and sid in ids[:3]:
                            from models.visualization.trajectory_gallery import write_compact_gallery
                            write_compact_gallery(out/'examples'/domain/mid/digest(sid)[:20],sid,image,gt,paths,
                                                  gt_masks=masks,pred_masks=pm,metadata=metrics)
                        write_json(cache,record)
                        if cursor%10==0: print(mid,domain,f'{cursor+1}/{len(ids)}',flush=True)
                    except Exception:
                        with (out/'failures.jsonl').open('a') as f:
                            f.write(json.dumps(dict(model_id=mid,domain=domain,sample_id=sid,smoke=smoke,error=traceback.format_exc()))+'\n')
                        raise
                completed[mid+'/'+domain]=len(ids)
                write_json(out/('smoke/status.json' if smoke else 'status.json'),dict(state='running',completed=completed))
            finally: catalog.close()
        del stroke,point; torch.cuda.empty_cache()
    if not smoke: completed=collect(out,cfg)
    write_json(out/('smoke/status.json' if smoke else 'status.json'),dict(state='complete',completed=completed,
               config_sha256=sha256_file(out/'config.json'),full_split=False,evaluation_scope='test_subset'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','smoke','run','aggregate']); p.add_argument('--output',required=True)
    args=p.parse_args(); out=Path(args.output).resolve()
    if args.mode=='prepare': prepare(out)
    elif args.mode=='aggregate':
        cfg=validate_frozen(out);cfg['_output']=str(out); collect(out,cfg)
    else: run(out,args.mode=='smoke')

if __name__=='__main__': main()
