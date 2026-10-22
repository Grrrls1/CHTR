"""Independent three-task AR training: predicted geometry, target-length unfolding."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from models.datasets.hierarchical_ar_dataset import TrajectoryDataset
from models.losses.hierarchical_ar_loss import three_task_loss,geometry_metrics,sample_lengths
from models.networks.hierarchical_ar import MODE,VERSION
from models.training.hierarchical_ar_setup import local_path,read_checkpoint,create_model,training_signature,write_json
from models.training.hierarchical_ar_evaluation import evaluate,to_device
from models.training.early_stopping import EarlyStopping


def fixed_coverage(indices,lengths,limit,ids):
    ordered=sorted(indices,key=lambda i:(len(lengths[i]),ids[i]))
    if not ordered:return []
    return [ordered[int(i)] for i in np.linspace(0,len(ordered)-1,min(limit,len(ordered)),dtype=int)]


def split_dataset(dataset,lengths,config):
    n=len(dataset);fraction=config['validation_fraction']
    heldout_count=min(n-1,max(1,int(math.ceil(n*fraction)))) if fraction else 0
    order=torch.randperm(n,generator=torch.Generator().manual_seed(config['seed'])).tolist()
    heldout=sorted(order[:heldout_count]);train=sorted(order[heldout_count:])
    if config['train_limit']:train=fixed_coverage(train,lengths,config['train_limit'],dataset.ids)
    train_eval=fixed_coverage(train,lengths,config['eval_limit'],dataset.ids)
    heldout_eval=fixed_coverage(heldout,lengths,config['eval_limit'],dataset.ids)
    return train,heldout,train_eval,heldout_eval


def validate_config(config):
    if config.get('training_mode')!=MODE:raise ValueError('requires hierarchical_ar_three_task mode')
    if config.get('initialization')!='random' or config.get('initial_weights'):
        raise ValueError('new three-task runs require random initialization; use explicit --resume only')
    for name in ('batch_size','gradient_accumulation','epochs','threads','checkpoint_every','eval_every_epochs','eval_limit'):
        if type(config[name]) is not int or config[name]<1:raise ValueError(name+' must be a positive integer')
    for name in ('max_steps','train_limit'):
        if type(config[name]) is not int or config[name]<0:raise ValueError(name+' must be nonnegative')
    if not math.isfinite(config['validation_fraction']) or not 0<=config['validation_fraction']<1:
        raise ValueError('validation_fraction must be in [0,1)')
    for name in ('learning_rate','grad_clip'):
        if not math.isfinite(config[name]) or config[name]<=0:raise ValueError(name+' must be finite and positive')
    if not math.isfinite(config.get('weight_decay',0)) or config.get('weight_decay',0)<0:raise ValueError('invalid weight_decay')
    if set(config['loss_weights'])!={'geo','point_end','stroke_end'} or any(not math.isfinite(v) or v<0 for v in config['loss_weights'].values()):
        raise ValueError('requires exactly three finite nonnegative loss weights')
    if not any(config['loss_weights'].values()):raise ValueError('at least one loss weight must be positive')
    if set(config['decode'])!={'point_end_threshold','stroke_end_threshold'} or any(not math.isfinite(v) or not 0<v<1 for v in config['decode'].values()):
        raise ValueError('requires two stop thresholds in (0,1)')
    if config.get('normalization')!='xy_size_minus_one_width_size':raise ValueError('unsupported coordinate normalization')
    if config.get('loss_reduction')!='equal_images_equal_strokes_balanced_end_groups':raise ValueError('unsupported loss reduction')


def grad_norm(parameters):
    values=[p.grad.detach().float().norm().square() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def train_update(model,dataset,indices,device,config):
    log={};batch_size=config['batch_size'];total=len(indices)
    for start in range(0,total,batch_size):
        samples=[to_device(dataset[i],device) for i in indices[start:start+batch_size]]
        lengths=[sample_lengths(s) for s in samples]
        outputs=model(torch.stack([s['image'] for s in samples]),lengths)
        loss,terms=three_task_loss(outputs,samples,config['loss_weights'])
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite three-task loss')
        factor=len(samples)/total
        (loss*factor).backward()
        values={k:float(v.detach()) for k,v in terms.items()}
        values.update(geometry_metrics(outputs,samples))
        values.update(target_points=sum(sum(row) for row in lengths)/len(samples),
                      target_strokes=sum(map(len,lengths))/len(samples))
        for key,value in values.items():log[key]=log.get(key,0.)+value*factor
        del loss,terms,outputs,samples
    log['loss']=sum(config['loss_weights'][k]*log[k] for k in config['loss_weights'])
    for key,weight in config['loss_weights'].items():log['weighted_'+key]=weight*log[key]
    return log


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/hierarchical_ar.json'))
    for name in ('output-dir','device','dataset','resume'):parser.add_argument('--'+name)
    for name in ('batch-size','epochs','max-steps','train-limit','eval-limit'):parser.add_argument('--'+name,type=int)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    for key in ('output_dir','device','dataset','batch_size','epochs','max_steps','train_limit','eval_limit'):
        if getattr(args,key) is not None:config[key]=getattr(args,key)
    config['dataset']=str(local_path(config['dataset']));config['output_dir']=str(local_path(config['output_dir']))
    validate_config(config)
    torch.set_num_threads(config['threads']);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=torch.device(config['device'])
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; submit GPU work through Slurm')
        torch.cuda.set_device(device);torch.cuda.manual_seed_all(config['seed'])
    checkpoint_path=local_path(args.resume) if args.resume else None
    ck=read_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    output=Path(config['output_dir'])
    if ck is None and output.exists() and any(output.iterdir()):raise ValueError('new run requires empty output directory')
    if ck is not None and output!=checkpoint_path.parent:raise ValueError('resume must use the original checkpoint directory')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.training.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('another trainer owns this output directory')
    dataset=TrajectoryDataset(config['dataset'])
    try:
        lengths,digest=dataset.inspect(config['model']['max_strokes'],config['model']['max_points'])
        training,heldout,train_eval,heldout_eval=split_dataset(dataset,lengths,config)
        split=dict(train_ids=[dataset.ids[i] for i in training],heldout_ids=[dataset.ids[i] for i in heldout],
                   train_eval_ids=[dataset.ids[i] for i in train_eval],heldout_eval_ids=[dataset.ids[i] for i in heldout_eval])
        signature=training_signature(config,digest,split)
        if ck is not None and ck.get('signature')!=signature:raise ValueError('resume data/split/training settings differ')
        model=create_model(config,device,ck)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config.get('weight_decay',0.))
        stopper=EarlyStopping(config['early_stopping'])
        step=epoch=cursor=0;best=-1.
        if ck is not None:
            optimizer.load_state_dict(ck['optimizer']);stopper.load_state_dict(ck['early_stopping'])
            step,epoch,cursor,best=ck['step'],ck['epoch'],ck['cursor'],ck['best_iou']
            torch.set_rng_state(ck['rng']['torch']);random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
            if device.type=='cuda' and ck['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        write_json(output/'config.json',config);write_json(output/'split.json',dict(**split,dataset_digest=digest))
        print(json.dumps(dict(mode=MODE,initialization='resume' if ck else 'random',parameters=sum(p.numel() for p in model.parameters()),
              batch_size=config['batch_size'],train_images=len(training),heldout_images=len(heldout),
              max_target_strokes=max(map(len,lengths)),max_target_points=max(max(row) for row in lengths),
              gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)),flush=True)
        stopped=[False]
        def request_stop(signum,frame):
            stopped[0]=True;print('stop requested; save after optimizer boundary',flush=True)
        for signum in (signal.SIGUSR1,signal.SIGTERM):signal.signal(signum,request_stop)
        def save(name):
            payload=dict(model_mode=MODE,model_version=VERSION,signature=signature,config=config,
                model=model.state_dict(),optimizer=optimizer.state_dict(),early_stopping=stopper.state_dict(),
                step=step,epoch=epoch,cursor=cursor,best_iou=best,
                rng=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                         cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None))
            temp=output/(name+'.tmp');torch.save(payload,temp);temp.replace(output/name)
        def status(reason='running'):
            write_json(output/'training_status.json',dict(step=step,completed_epochs=epoch,cursor=cursor,reason=reason,
                batch_size=config['batch_size'],effective_batch_size=config['batch_size']*config['gradient_accumulation'],
                train_samples=len(training),heldout_samples=len(heldout),target_epochs=config['epochs'],
                best_iou=best,best_epoch_loss=stopper.best,bad_epochs=stopper.bad_epochs,
                training_semantics='predicted geometry history; target-length unfolding',
                peak_memory_note='training logs distinguish allocated/reserved; caps do not guarantee capacity'))
        def validation():
            nonlocal best
            gallery=output/f'eval_step_{step:06d}'
            if gallery.exists():gallery=output/f'eval_step_{step:06d}_retry_{time.time_ns()}'
            train_report=evaluate(model,dataset,train_eval,device,config['decode'],gallery/'training_images','training_images')
            valid_report=evaluate(model,dataset,heldout_eval,device,config['decode'],gallery/'heldout_images','heldout_images') if heldout_eval else None
            report=dict(epoch=epoch,step=step,training_images=train_report,heldout_images=valid_report)
            write_json(gallery/'summary.json',report)
            with (output/'validation.jsonl').open('a') as stream:stream.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
            chosen=valid_report or train_report
            if chosen['mean_image_only_iou']>best:best=chosen['mean_image_only_iou'];save('best.pt')
            print(json.dumps(dict(eval_step=step,epoch=epoch,selection_split=chosen['split'],iou=chosen['mean_image_only_iou'])),flush=True)
        model.train()
        if ck is None:
            # A scheduler restart can resume even if initial evaluation is interrupted.
            save('last.pt')
            validation()
            save('last.pt')
        status()
        if stopper.should_stop:status('early_stopping');return
        last_eval_step=step if ck is None else None
        while not stopped[0] and epoch<config['epochs'] and (not config['max_steps'] or step<config['max_steps']):
            order=[training[i] for i in torch.randperm(len(training),generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()]
            while not stopped[0] and cursor<len(order) and (not config['max_steps'] or step<config['max_steps']):
                chunk=order[cursor:cursor+config['batch_size']*config['gradient_accumulation']]
                if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                started=time.monotonic();optimizer.zero_grad(set_to_none=True)
                log=train_update(model,dataset,chunk,device,config)
                for name in ('encoder','stroke_decoder','geometry_head','point_end_head','stroke_end_head'):
                    log[name+'_grad_norm']=grad_norm(getattr(model,name).parameters())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
                if not bool(torch.isfinite(norm)):raise RuntimeError('nonfinite three-task gradient')
                optimizer.step();step+=1;cursor+=len(chunk);stopper.add(log['loss'],len(chunk))
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                    log.update(peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
                               peak_cuda_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2)
                log.update(step=step,epoch=epoch,samples_in_update=len(chunk),batch_size=min(len(chunk),config['batch_size']),
                           step_seconds=time.monotonic()-started)
                with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(log,allow_nan=False)+'\n')
                print(json.dumps(log),flush=True)
                if step%config['checkpoint_every']==0:save('last.pt');status()
            if stopped[0]:save('last.pt');status('interrupted');return
            if cursor==len(order):
                epoch+=1;cursor=0;mean,improved,stop=stopper.finish_epoch(epoch)
                summary=dict(epoch=epoch,step=step,mean_training_loss=mean,best_loss=stopper.best,bad_epochs=stopper.bad_epochs)
                with (output/'epoch_metrics.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
                if improved:save('best_loss.pt')
                if epoch%config['eval_every_epochs']==0 or epoch==config['epochs'] or stop:
                    validation();last_eval_step=step;save(f'epoch_{epoch:03d}.pt')
                save('last.pt');status('early_stopping' if stop else 'running')
                if stop:return
        if stopped[0]:save('last.pt');status('interrupted');return
        if last_eval_step!=step:validation()
        save('last.pt');status('max_epochs' if epoch>=config['epochs'] else 'max_steps')
    finally:
        dataset.close();lock.close()


if __name__=='__main__':main()
