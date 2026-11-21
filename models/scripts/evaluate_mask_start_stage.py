"""E1 mask/start transmission and E2 stage crossing. Frozen checkpoints only."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import platform
import scipy
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.experiments import mask_start_core as c
from models.scripts import evaluate_cross_domain_sampled as old
from models.training.ordered_stroke import load_model as load_stroke
from models.training.single_stroke import load_model as load_point,make_catalog

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'models/runs/sampled_cross_domain_test/run_20261120_n100'
VERSION='mask_start_stage_cross_v1'


def prepare(out):
    if (out/'protocol.json').exists():raise ValueError('already prepared; never replace frozen protocol')
    out.mkdir(parents=True,exist_ok=True)
    base=c.read_json(BASE/'config.json');weights={m:base['weights']['M_'+m] for m in ['BOOK','TRANSFER']}
    for m,w in weights.items():
        for stage,cap in [('stroke','max_strokes'),('point','max_points')]:
            ref=w[stage];assert old.sha256_file(ref['path'])==ref['sha256']
            ck=torch.load(ref['path'],map_location='cpu',weights_only=False)
            assert ck['config']['model'][cap]==(64 if stage=='stroke' else 400)
            assert (ck['epoch'],ck['step'])==(ref['epoch'],ref['step']);del ck
    domains={}
    for domain,dc in base['domains'].items():
        split=c.read_json(ROOT/dc['config']['split_file'])
        assert old.sha256_file(ROOT/dc['config']['split_file'])==dc['split_sha256']
        ids=np.random.default_rng(20261120).choice(sorted(split['val_ids']),100,replace=False).tolist()
        # Fixed smoke coverage over stroke count and trajectory length, selected before model results.
        cat=make_catalog(dc['config'])
        try:
            def key(sid):
                row=cat.rows[sid]
                return (row['stroke_count'],max(b['point_range'][1]-b['point_range'][0] for b in row['stroke_map']),sid)
            ranked=sorted(ids,key=key);smoke=[ranked[i] for i in np.linspace(0,99,8,dtype=int)]
        finally:cat.close()
        domains[domain]=dict(config=dc['config'],fingerprint=dc['fingerprint'],split_sha256=dc['split_sha256'],
            render=dc['render'],candidate_count=dc['candidate_count'],val_ids=sorted(ids),smoke_ids=smoke,test_ids=dc['union_ids'])
        c.write_json(out/'manifests'/domain/'val100.json',dict(ids=sorted(ids),seed=20261120,split='val',selection='sorted val IDs, default_rng choice replace=False',numpy=np.__version__))
        for seed in base['sampling_seeds']:
            src=BASE/'manifests'/domain/f'seed_{seed}.json';dst=out/'manifests'/domain/src.name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            assert old.sha256_file(dst)==dc['manifest_sha256'][str(seed)]
        domains[domain]['test_manifest_sha256']=dc['manifest_sha256']
        domains[domain]['val_manifest_sha256']=old.sha256_file(out/'manifests'/domain/'val100.json')
    protocol=dict(runtime_versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__),version=VERSION,weights=weights,domains=domains,global_seed=20261120,noise_draws=5,
        noise_levels=list(c.LEVELS),point_batch_size=16,max_strokes=64,max_points=400,threshold=.5,
        precision='FP32',tf32=False,matching=dict(threshold=.5,objective='maximum valid cardinality, then sum IoU',
            tie_break='canonical pred-index/GT-index ordered matrix; recorded scipy linear_sum_assignment deterministic tie handling'),
        noise_seed='SHA256 compact JSON [global_seed,sample_id,pred_index,draw,component]; first 16 bytes little endian -> default_rng.standard_normal',
        clipping=dict(xy_px=[0,255],w_min_px=.001,w_max=None),response_protocol='mask_start_noise_response_arc64_v2',
        arc_degeneracy='singleton/zero-length: repeat first XYW; report degenerate flags and length, never flip direction',
        baseline_config_sha256=old.sha256_file(BASE/'config.json'),repeats=base['sampling_seeds'],
        numerical_metrics='same GT-XYW raster, Otsu image reference, exact CPU DTW as original baseline',
        timing='E2 sum of synchronized image-transfer/stroke and point components; excludes cache IO and preloading cached tensors; not an independent contiguous timing repeat',
        base_old_predictions='fresh E2 execution; historical BB/TT only reproducibility targets',
        conditions= c.conditions())
    c.write_json(out/'protocol.json',protocol)
    rev=old.source_revision();c.write_json(out/'code_revision.json',rev)
    sources=out/'source';sources.mkdir(exist_ok=True)
    for rel in rev['source_sha256']:
        dst=sources/(rel.replace('/','__')+'.txt');shutil.copyfile(ROOT/rel,dst)
    c.write_csv(out/'condition_manifest.csv',protocol['conditions'])
    execution=[]
    for split in ['val','test']:
        for cond in protocol['conditions']:
            for draw in range(cond['draws']):
                execution.append(dict(**cond,split=split,draw=draw,unique_characters=len(domains[cond['domain']][split+'_ids']),
                                      state='planned',completed=0,cache_reuse='E1 PRED/PRED derives from E2 S_TRANSFER'))
    assert len(execution)==396
    c.write_csv(out/'execution_manifest.csv',execution)
    c.write_json(out/'status.json',dict(state='prepared',conditions=54,units_per_formal_split=198,formal_execution_rows=396))
    (out/'failures.jsonl').touch()
    (out/'protocol.json').chmod(0o444);(out/'code_revision.json').chmod(0o444)
    print('prepared',out,flush=True)


class Experiment:
    def __init__(self,out,gpu=True):
        self.out=out;self.cfg=c.read_json(out/'protocol.json');self.cats={};self.gpu=gpu
        assert self.cfg['runtime_versions']==dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__),'runtime version changed'
        rev=c.read_json(out/'code_revision.json')
        for rel,h in rev['source_sha256'].items():
            if old.sha256_file(ROOT/rel)!=h:raise ValueError('frozen source changed: '+rel)
        for w in self.cfg['weights'].values():
            for stage in ['stroke','point']:assert old.sha256_file(w[stage]['path'])==w[stage]['sha256']
        for domain,d in self.cfg['domains'].items():
            assert old.sha256_file(out/'manifests'/domain/'val100.json')==d['val_manifest_sha256']
            for seed,h in d['test_manifest_sha256'].items():
                assert old.sha256_file(out/'manifests'/domain/f'seed_{seed}.json')==h
        self.identity=c.digest(dict(protocol=self.cfg,source=rev['source_sha256']))
        if gpu:
            assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'all CUDA must use Slurm'
            torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            cv2.setNumThreads(4)
            self.env=dict(job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
                gpu_properties=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,cuda=torch.version.cuda,
                numpy=np.__version__,scipy=scipy.__version__,precision='FP32',tf32=False,point_batch_size=16)
            c.write_json(out/'environments'/f"job_{self.env['job_id']}.json",self.env)
        self.stroke=None;self.stroke_id=None;self.point=None;self.point_id=None

    def close(self):
        for cat in self.cats.values():cat.close()

    def catalog(self,domain):
        if domain not in self.cats:
            d=self.cfg['domains'][domain]
            assert old.sha256_file(ROOT/d['config']['split_file'])==d['split_sha256']
            cat=make_catalog(d['config']);assert cat.fingerprint==d['fingerprint'];self.cats[domain]=cat
        return self.cats[domain]

    def ids(self,split,domain):return self.cfg['domains'][domain][('smoke' if split=='smoke' else split)+'_ids']

    def assets(self,domain,sid):
        image,points,offsets,masks=self.catalog(domain).character_assets(sid)
        return image,[points[a:b] for a,b in zip(offsets[:-1],offsets[1:])],masks

    def commit(self,path,record):
        record.update(committed=True,identity=self.identity,job_id=self.env['job_id'])
        c.write_json(path,record)

    def cached(self,path):
        try:r=c.read_json(path)
        except (FileNotFoundError,json.JSONDecodeError):return None
        return r if r.get('committed') is True and r.get('identity')==self.identity else None

    def stage_path(self,split,domain,stage,sid):return self.out/'instances'/split/domain/stage/(c.digest(sid)+'.json')
    def e2_path(self,split,domain,stage,point,sid):return self.out/'E2'/split/domain/(stage+'_'+point)/(c.digest(sid)+'.json')
    def e1_path(self,split,cond,draw,sid):return self.out/cond['experiment']/split/cond['condition_id']/f'draw_{draw}'/(c.digest(sid)+'.json')

    def load_stage(self,split,domain,stage,sid):
        path=self.stage_path(split,domain,stage,sid);meta=self.cached(path)
        if meta is None:raise ValueError('missing stage cache: '+str(path))
        npz=path.with_suffix('.npz');assert old.sha256_file(npz)==meta['npz_sha256']
        with np.load(npz,allow_pickle=False) as z:
            masks=np.unpackbits(z['mask_bits'],axis=-1,count=256).astype(bool)
            return dict(masks=masks,starts=z['starts'].copy(),boxes=z['boxes'].copy(),confidences=z['confidences'].copy(),
                        terminated=meta['terminated'],truncated=meta['truncated'],stop_reason=meta['stop_reason'],meta=meta)

    def model(self,kind,key):
        if kind=='stroke':
            if self.stroke_id!=key:
                self.stroke=None;torch.cuda.empty_cache()
                self.stroke,ck=load_stroke(self.cfg['weights'][key]['stroke']['path'],'cuda:0');del ck
                self.stroke.eval();self.stroke_id=key
            return self.stroke
        if self.point_id!=key:
            self.point=None;torch.cuda.empty_cache()
            self.point,ck=load_point(self.cfg['weights'][key]['point']['path'],'cuda:0');del ck
            self.point.eval();self.point_id=key
        return self.point

    def generate_stages(self,split):
        for stage in ['BOOK','TRANSFER']:
            model=self.model('stroke',stage)
            for domain in ['BOOK','CASIA']:
                # Validation image warmup only, also before test execution.
                warm=self.assets(domain,self.cfg['domains'][domain]['smoke_ids'][0])[0]
                with torch.inference_mode():model.generate(torch.from_numpy(warm.astype(np.float32)[None,None]/255).cuda(),max_strokes=64,threshold=.5)
                for index,sid in enumerate(self.ids(split,domain)):
                    path=self.stage_path(split,domain,stage,sid)
                    if self.cached(path):
                        self.load_stage(split,domain,stage,sid);continue
                    image,_,_=self.assets(domain,sid)
                    torch.cuda.synchronize();t=time.perf_counter()
                    x=torch.from_numpy(image.astype(np.float32)[None,None]/255).cuda()
                    with torch.inference_mode():r=model.generate(x,max_strokes=64,threshold=.5)[0]
                    torch.cuda.synchronize();seconds=time.perf_counter()-t
                    masks=(r['masks']>=.5).cpu().numpy()
                    b=io.BytesIO();np.savez_compressed(b,mask_bits=np.packbits(masks,axis=-1),starts=r['starts'].cpu().numpy(),
                        boxes=r['boxes'].cpu().numpy(),confidences=r['confidences'].cpu().numpy())
                    c.atomic_bytes(path.with_suffix('.npz'),b.getvalue())
                    self.commit(path,dict(sample_id=sid,domain=domain,stage=stage,terminated=bool(r['terminated']),
                        truncated=bool(r['truncated']),stop_reason=r['stop_reason'],image_transfer_and_stroke_seconds=seconds,
                        npz_sha256=old.sha256_file(path.with_suffix('.npz')),mask_encoding='threshold .5, uint8 packbits last axis'))
                    if index%10==0:print('instances',split,domain,stage,index+1,len(self.ids(split,domain)),flush=True)
        self.stroke=None;self.stroke_id=None;torch.cuda.empty_cache()

    def decode(self,point,image,masks,starts):
        if not len(starts):return [],[],0.,dict(peak_allocated_bytes=torch.cuda.memory_allocated(),peak_reserved_bytes=torch.cuda.memory_reserved())
        model=self.model('point',point)
        x=torch.from_numpy(image.astype(np.float32)[None]/255).cuda()
        mt=torch.from_numpy(np.asarray(masks,bool)).cuda();st=torch.from_numpy(np.asarray(starts,np.float32)).cuda()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();result=[]
        with torch.inference_mode():
            for offset in range(0,len(st),16):
                mask=mt[offset:offset+16];first=st[offset:offset+16]
                inputs=torch.where(mask[:,None],x[None],torch.ones_like(x)[None])
                generated=model.generate(inputs,first,max_points=400,threshold=.5)
                for initial,r in zip(first,generated):
                    assert torch.equal(r['points'][0],initial),'point model rewrote conditioned first point'
                result.extend(generated)
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        peaks=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        paths=[r['points'].cpu().numpy()*c.SCALE for r in result]
        states=[dict(terminated=bool(r['terminated']),truncated=bool(r['truncated']),stop_reason=r['stop_reason'],
                     next_validity=r['next_validity'],first_point_exact=True) for r in result]
        return paths,states,seconds,peaks

    def warm_point(self,point):
        image,gt,masks=self.assets('BOOK',self.cfg['domains']['BOOK']['smoke_ids'][0])
        for _ in range(2):self.decode(point,image,masks[:min(16,len(gt))],np.asarray([p[0] for p in gt[:16]],np.float32)/c.SCALE)

    def run_e2(self,split):
        for point in ['BOOK','TRANSFER']:
            self.model('point',point);self.warm_point(point)
            for domain in ['BOOK','CASIA']:
                for stage in ['BOOK','TRANSFER']:
                    for index,sid in enumerate(self.ids(split,domain)):
                        path=self.e2_path(split,domain,stage,point,sid)
                        if self.cached(path):continue
                        image,gt,gmask=self.assets(domain,sid);inst=self.load_stage(split,domain,stage,sid)
                        paths,states,secs,peaks=self.decode(point,image,inst['masks'],inst['starts'])
                        t=time.perf_counter();metrics=c.score_character(image,gt,gmask,paths,states,inst)
                        metrics.update(inference_seconds=secs+inst['meta']['image_transfer_and_stroke_seconds'],**peaks)
                        self.commit(path,dict(sample_id=sid,split=split,domain=domain,stage=stage,point=point,
                            stage_npz_sha256=inst['meta']['npz_sha256'],prediction=c.pack_prediction(paths,states),metrics=metrics,
                            point_gpu_seconds=secs,metric_cpu_seconds=time.perf_counter()-t,first_point_exact=True))
                        if index%10==0:print('E2',split,domain,stage,point,index+1,len(self.ids(split,domain)),flush=True)

    def match(self,split,sid):
        path=self.out/'matches'/split/(c.digest(sid)+'.json')
        r=self.cached(path)
        if r:return r
        image,gt,gmask=self.assets('BOOK',sid);inst=self.load_stage(split,'BOOK','TRANSFER',sid)
        pairs,_=c.matching(inst['masks'],gmask)
        for pair in pairs:
            pi,gi=pair['pred_index'],pair['gt_index'];pm=inst['masks'][pi];gm=gmask[gi];inter=int((pm&gm).sum())
            actual=inst['starts'][pi]*c.SCALE;truth=gt[gi][0]
            inside=lambda xy:bool(pm[int(np.clip(round(float(xy[1])),0,255)),int(np.clip(round(float(xy[0])),0,255))])
            pair.update(mask_precision=inter/max(int(pm.sum()),1),mask_recall=inter/max(int(gm.sum()),1),
                mask_area_ratio=float(pm.sum()/max(gm.sum(),1)),pred_start_inside_pred_mask=inside(actual),
                gt_start_inside_pred_mask=inside(truth),gt_first_xyw_px=truth.tolist(),pred_first_xyw_px=actual.tolist())
        r=dict(sample_id=sid,stage_npz_sha256=inst['meta']['npz_sha256'],pairs=pairs,gt_strokes=len(gt),pred_strokes=len(inst['starts']),
               unmatched_pred=sorted(set(range(len(inst['starts'])))-{v['pred_index'] for v in pairs}),
               unmatched_gt=sorted(set(range(len(gt)))-{v['gt_index'] for v in pairs}))
        self.commit(path,r);return r

    def calibration(self,split):
        path=self.out/('smoke_noise_calibration.json' if split=='smoke' else 'noise_calibration.json')
        if path.exists():return c.read_json(path)
        residual=[]
        for sid in self.ids(split,'BOOK'):
            for m in self.match(split,sid)['pairs']:
                residual.append(np.asarray(m['pred_first_xyw_px'])-np.asarray(m['gt_first_xyw_px']))
        cal=c.calibrate(residual);cal.update(identity=self.identity,calibration_split='val',calibration_ids=self.ids(split,'BOOK'))
        c.write_json(path,cal);path.chmod(0o444);return cal

    def run_condition(self,split,cond,draw=0,cal=None):
        point=cond['point'];self.model('point',point)
        for cursor,sid in enumerate(self.ids(split,'BOOK')):
            path=self.e1_path(split,cond,draw,sid)
            cached=self.cached(path)
            noise_hash=c.digest(cal) if cond['mode'] in ['N10','N01','N11'] else None
            if cached and cached.get('noise_calibration_sha256')==noise_hash:continue
            image,gt,gmask=self.assets('BOOK',sid);inst=self.load_stage(split,'BOOK','TRANSFER',sid)
            match=self.match(split,sid);base=self.cached(self.e2_path(split,'BOOK','TRANSFER',point,sid))
            if base is None:raise ValueError('E1 requires committed E2 reference')
            base_paths=c.paths_from(base);base_states=base['prediction']['point_states']
            if cond['experiment']=='GT_REF':
                first=np.asarray([p[0] for p in gt],np.float32)/c.SCALE
                generated,states,secs,peaks=self.decode(point,image,gmask,first)
                metric_started=time.perf_counter()
                matched=[]
                for gi,pred in enumerate(generated):
                    r=c.instance_metrics(gt[gi],pred,pred,[0,0,0],states[gi]);r.update(gt_index=gi,pred_index=None);matched.append(r)
                full_inst=dict(masks=gmask,terminated=True,truncated=False)
                full=c.score_character(image,gt,gmask,generated,states,full_inst)
                coverage=dict(gt_strokes=len(gt),pred_strokes=len(gt),matched=len(gt),all_gt_reference=True)
            else:
                pairs=match['pairs'];inputs=[];first=[];noise=[]
                for pair in pairs:
                    pi,gi=pair['pred_index'],pair['gt_index']
                    start,metadata=c.first_point(gt[gi][0],inst['starts'][pi]*c.SCALE,cond['mode'],cond['k'],draw,
                        cal or dict(sigma_xy_base_px=0,sigma_w_base_px=0),self.cfg['global_seed'],sid,pi)
                    first.append(start/c.SCALE);noise.append(metadata)
                    inputs.append(gmask[gi] if cond['mask']=='GT' else inst['masks'][pi])
                if cond['mask']=='PRED' and cond['mode']=='PRED':
                    generated=[base_paths[p['pred_index']] for p in pairs];states=[base_states[p['pred_index']] for p in pairs]
                    secs=None;peaks={k:base['metrics'][k] for k in ['peak_allocated_bytes','peak_reserved_bytes']}
                else:
                    generated,states,secs,peaks=self.decode(point,image,np.asarray(inputs,bool),np.asarray(first,np.float32).reshape(-1,3))
                full_paths=list(base_paths);full_states=list(base_states);matched=[]
                clean=None
                if cond['mode'] in ['N10','N01','N11','PRED']:
                    clean_cond=dict(cond,mode='N00',k=0.,draws=1,condition_id=f"E1_{point}_{cond['mask']}_N00")
                    clean=self.cached(self.e1_path(split,clean_cond,0,sid))
                    if clean is None:raise ValueError('same-mask N00 must precede noise/PRED response')
                    clean_paths=c.paths_from(clean)
                metric_started=time.perf_counter()
                for pair,pred,state,nm in zip(pairs,generated,states,noise):
                    pi,gi=pair['pred_index'],pair['gt_index'];clean_path=pred if clean is None else clean_paths[pi]
                    delta=np.asarray(nm['actual_noise_px'])
                    clean_state=state if clean is None else clean['prediction']['point_states'][pi]
                    r=c.instance_metrics(gt[gi],pred,clean_path,delta,state,clean_state)
                    r.update(**pair,noise=nm,input_first_xyw_px=(np.asarray(first[len(matched)])*c.SCALE).tolist(),first_point_exact=True)
                    matched.append(r);full_paths[pi]=pred;full_states[pi]=state
                generated=full_paths;states=full_states
                full=c.score_character(image,gt,gmask,generated,states,inst)
                coverage=dict(gt_strokes=match['gt_strokes'],pred_strokes=match['pred_strokes'],matched=len(pairs),
                    unmatched_pred=len(match['unmatched_pred']),unmatched_gt=len(match['unmatched_gt']),
                    matching_identity=c.digest(match),stage_npz_sha256=inst['meta']['npz_sha256'])
            full.update(inference_seconds=None,**peaks)
            self.commit(path,dict(sample_id=sid,split=split,condition=cond,draw=draw,prediction=c.pack_prediction(generated,states),
                whole_character=full,instances=matched,single_character=c.single_character_metrics(matched),coverage=coverage,
                conditional_point_gpu_seconds=secs,metric_cpu_seconds=time.perf_counter()-metric_started,noise_calibration_sha256=noise_hash,first_point_exact=True,
                deployment_condition=cond['experiment']=='E1' and cond['mask']=='PRED' and cond['mode']=='PRED'))
            if cursor%10==0:
                print(cond['condition_id'],split,'draw',draw,cursor+1,len(self.ids(split,'BOOK')),flush=True)
                c.write_json(self.out/'progress'/f"job_{self.env['job_id']}.json",dict(split=split,condition_id=cond['condition_id'],draw=draw,completed=cursor+1,required=len(self.ids(split,'BOOK'))))

    def basic(self,split):
        self.generate_stages(split);self.run_e2(split)
        conditions=[x for x in self.cfg['conditions'] if x['experiment']!='E2' and x['draws']==1]
        # N00 precedes PRED within each point/mask; all-GT reference remains a distinct dataset.
        for cond in conditions:self.run_condition(split,cond)

    def accept_basic(self,split):
        count=0
        for cond in self.cfg['conditions']:
            if cond['draws']!=1:continue
            for sid in self.ids(split,cond['domain']):
                path=self.e2_path(split,cond['domain'],cond['stroke'],cond['point'],sid) if cond['experiment']=='E2' else self.e1_path(split,cond,0,sid)
                record=self.cached(path);assert record and record['first_point_exact']
                if cond['experiment']=='E2':
                    same=self.load_stage(split,cond['domain'],cond['stroke'],sid)
                    assert record['stage_npz_sha256']==same['meta']['npz_sha256']
                    assert len(record['prediction']['stroke_offsets'])-1==len(same['starts'])
                else:
                    assert len(record['instances'])==record['coverage']['matched']
                    if cond['experiment']=='E1':
                        assert record['coverage']['matching_identity']==c.digest(self.match(split,sid))
                        if cond['mask']=='PRED' and cond['mode']=='PRED':
                            base=self.cached(self.e2_path(split,'BOOK','TRANSFER',cond['point'],sid))
                            assert record['prediction']==base['prediction']
                count+=1
        c.write_json(self.out/'acceptance'/f'{split}_basic.json',dict(state='passed',records=count,conditions=18,
            first_point_preserved=True,matching_fixed=True,E2_stage_shared=True,O11_equals_E2=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','smoke','base','noise'])
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['val','test']);p.add_argument('--partition',type=int)
    a=p.parse_args();out=Path(a.output).resolve()
    if a.mode=='prepare':prepare(out);return
    ex=Experiment(out)
    try:
        c.write_json(out/'status.json',dict(state='running',mode=a.mode,split=a.split,partition=a.partition,job_id=ex.env['job_id']))
        if a.mode=='smoke':
            started=time.monotonic();ex.basic('smoke');ex.accept_basic('smoke');cal=ex.calibration('smoke')
            for cond in ex.cfg['conditions']:
                if cond['draws']==5:ex.run_condition('smoke',cond,0,cal)
            from models.experiments.mask_start_summary import smoke_report
            smoke_report(ex)
            c.write_json(out/'acceptance/smoke.json',dict(state='passed',conditions=54,noise_draws=1,
                elapsed_seconds=time.monotonic()-started,book_characters=8,casia_characters=8))
        elif a.mode=='base':
            assert c.read_json(out/'acceptance/smoke.json')['state']=='passed'
            ex.basic('val');ex.accept_basic('val');ex.calibration('val')
            overview=[]
            for sid in ex.ids('val','BOOK'):
                m=ex.match('val',sid);overview.append(dict(sample_id=sid,matched=len(m['pairs']),gt_strokes=m['gt_strokes'],pred_strokes=m['pred_strokes']))
            c.write_json(out/'val_basic_overview.json',dict(characters=100,instances=overview,
                gt_match_coverage=sum(r['matched'] for r in overview)/sum(r['gt_strokes'] for r in overview),
                source='fixed val100; no threshold or sampling adjustment after results'))
            ex.basic('test');ex.accept_basic('test')
            from models.experiments.mask_start_summary import aggregate
            aggregate(out,'base')
        else:
            assert a.partition is not None and 0<=a.partition<12 and a.split in ['val','test']
            assert c.read_json(out/'acceptance/test_basic.json')['state']=='passed'
            if a.split=='test':assert c.read_json(out/'acceptance/val_noise.json')['state']=='passed'
            cal=c.read_json(out/'noise_calibration.json');assert cal['identity']==ex.identity
            point=['BOOK','TRANSFER'][a.partition//6];mask=['GT','PRED'][(a.partition//3)%2];k=c.LEVELS[a.partition%3]
            selected=[x for x in ex.cfg['conditions'] if x['draws']==5 and x['point']==point and x['mask']==mask and x['k']==k]
            ex.model('point',point);ex.warm_point(point)
            for cond in selected:
                for draw in range(5):ex.run_condition(a.split,cond,draw,cal)
            c.write_json(out/'acceptance'/f'{a.split}_noise_partition_{a.partition}.json',dict(state='passed',conditions=3,units=15,
                characters=len(ex.ids(a.split,'BOOK')),calibration_sha256=c.digest(cal)))
    except Exception:
        with (out/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(mode=a.mode,split=a.split,partition=a.partition,error=traceback.format_exc()))+'\n')
        raise
    finally:ex.close()

if __name__=='__main__':main()
