"""Full-dataset free rollout with periodic checkpoint and visual evaluation."""
import argparse
import hashlib
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent/'reproduction/pipeline'))
from common import render, states, iou
from models.datasets.lmdb_dataset import CHTRModelDataset
from models.networks.model import StrokeTrajectoryModel
from models.networks.free_rollout import rollout, visual_predictions
from models.losses.free_rollout_loss import trajectory_loss, visual_loss, weighted_loss, DEFAULT_WEIGHTS


def local_path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def to_device(sample, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}


@torch.no_grad()
def evaluate(model, dataset, indices, device, config):
    model.eval()
    rows = []
    for index in indices:
        s = to_device(dataset[index], device)
        r = rollout(model, s['image'][None], **config['rollout'])
        chunks = [v['geometry'].cpu().numpy() for v in r['strokes']]
        off = np.r_[0, np.cumsum([len(v) for v in chunks])]
        points = np.c_[np.concatenate(chunks), np.zeros((int(off[-1]), 3), np.float32)]
        size = s['image_size']; points[:, :2] *= size-1; points[:, 2] *= size
        # For display/IoU only: states() imposes endpoints on a copy, including
        # truncated outputs. Actual rollout termination remains recorded below.
        image = render(states(points.copy(), off), off, size=size)
        truth = s['image'][0].cpu().numpy() < .5
        terms = trajectory_loss(r, s, config)
        gt = len(s['stroke_types']); pred = len(chunks)
        rows.append(dict(sample_id=s['sample_id'], character=s['character'],
                         image_only_iou=iou(image >= 128, truth), gt_strokes=gt,
                         predicted_strokes=pred, completed=r['completed'],
                         truncated=r['truncated'], truncation_reason=r['truncation_reason'],
                         **{k:float(v) for k,v in terms.items()}))
    model.train()
    return dict(mean_image_only_iou=float(np.mean([r['image_only_iou'] for r in rows])),
                exact_stroke_count=sum(r['gt_strokes']==r['predicted_strokes'] for r in rows),
                completed=sum(r['completed'] for r in rows), truncated=sum(r['truncated'] for r in rows),
                validation_note='fixed training images; this is memorization evaluation, not generalization',
                rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/free_rollout.json'))
    p.add_argument('--output-dir'); p.add_argument('--device'); p.add_argument('--dataset')
    p.add_argument('--max-steps', type=int); p.add_argument('--epochs', type=int)
    init = p.add_mutually_exclusive_group()
    init.add_argument('--init-weights'); init.add_argument('--resume')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    for key in ('output_dir','device','dataset','max_steps','epochs'):
        if getattr(args,key) is not None: config[key] = getattr(args,key)
    if config.get('training_mode') != 'free_rollout': raise ValueError('requires free_rollout mode')
    if config['epochs'] < 1 or config['gradient_accumulation'] < 1 or config['checkpoint_every'] < 1:
        raise ValueError('epochs, accumulation and checkpoint_every must be positive')
    if config['max_steps'] < 0: raise ValueError('max_steps must be nonnegative')
    if config['train_limit'] < 0 or config['validation_limit'] < 1:
        raise ValueError('train_limit=0 means all; validation_limit must be positive')
    device = torch.device(config['device']); torch.set_num_threads(config.get('threads',2))
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(device)
    checkpoint_path = local_path(args.resume or args.init_weights or config['initial_weights'])
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    architecture = ck['config']['model']
    model = StrokeTrajectoryModel(**architecture).to(device)
    model.load_state_dict(ck['model'])
    ds = CHTRModelDataset(local_path(config['dataset']))
    if len(ds) < 1: raise ValueError('requires a nonempty ready dataset')
    # Stable stroke-count coverage selects fixed training-image evaluation,
    # but neither lengths nor labels are supplied to rollout().
    with ds._env().begin() as txn:
        ordered = sorted(range(len(ds)), key=lambda i:(json.loads(txn.get(ds.ids[i].encode()))['stroke_count'],ds.ids[i]))
    eval_indices = [ordered[int(i)] for i in np.linspace(0,len(ordered)-1,min(config['validation_limit'],len(ds)),dtype=int)]
    # Evaluation images remain in training: use the complete ready dataset.
    training = list(ordered)
    if config['train_limit'] and config['train_limit'] < len(training):
        training = [training[int(i)] for i in np.linspace(0,len(training)-1,config['train_limit'],dtype=int)]
    if config.get('require_full_dataset',True) and len(training) != len(ds):
        raise ValueError('full dataset run must use every ready sample')
    if not isinstance(config.get('eval_every_epochs',20),int) or config.get('eval_every_epochs',20) < 1:
        raise ValueError('eval_every_epochs must be positive')
    config['model'] = architecture
    config['initial_weights'] = str(local_path(config['initial_weights'])) if not args.init_weights else str(checkpoint_path)
    signature_config = {k:v for k,v in config.items() if k not in ('output_dir','device','epochs','max_steps','initial_weights')}
    signature = hashlib.sha256(json.dumps(dict(config=signature_config, dataset=str(local_path(config['dataset'])),
                  train_ids=[ds.ids[i] for i in training], eval_ids=[ds.ids[i] for i in eval_indices]), sort_keys=True).encode()).hexdigest()
    output = local_path(config['output_dir'])
    if args.resume:
        if ck.get('free_rollout_version') != 1 or ck.get('signature') != signature:
            raise ValueError('resume requires matching free-rollout version, data and training settings')
        if output != checkpoint_path.parent: raise ValueError('resume must use checkpoint output directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('new run requires an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (output/'split.json').write_text(json.dumps(dict(train_ids=[ds.ids[i] for i in training],
         validation_ids=[ds.ids[i] for i in eval_indices], evaluation_is_training_subset=True),indent=2)+'\n')
    groups = model.branch_parameter_groups() if model.separate_branches else [{'params':list(model.parameters()),'name':'joint'}]
    optimizer = torch.optim.AdamW(groups,lr=config['learning_rate'],weight_decay=config.get('weight_decay',0.))
    step=epoch=cursor=0; best=-1.
    if args.resume:
        optimizer.load_state_dict(ck['optimizer']);step=ck['step'];epoch=ck['epoch'];cursor=ck['cursor'];best=ck['best_iou']
        torch.set_rng_state(ck['rng']['torch']); random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
        if device.type=='cuda' and ck['rng']['cuda'] is not None: torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    def save(name):
        payload=dict(free_rollout_version=1,signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
            config=config,step=step,epoch=epoch,cursor=cursor,best_iou=best,
            rng=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        temp=output/(name+'.tmp');torch.save(payload,temp);temp.replace(output/name)
    def progress(reason='running'):
        target=output/'training_status.json';temp=output/'training_status.json.tmp'
        temp.write_text(json.dumps(dict(step=step,completed_epochs=epoch,cursor=cursor,
            train_samples=len(training),evaluation_samples=len(eval_indices),target_epochs=config['epochs'],
            eval_every_epochs=config.get('eval_every_epochs',20),best_iou=best,reason=reason),indent=2)+'\n')
        temp.replace(target)
    def validation():
        nonlocal best
        report=evaluate(model,ds,eval_indices,device,config);report.update(step=step,epoch=epoch)
        with (output/'validation.jsonl').open('a') as f:f.write(json.dumps(report,ensure_ascii=False)+'\n')
        if report['mean_image_only_iou'] > best:
            best=report['mean_image_only_iou'];save('best.pt')
        print(json.dumps(dict(validation_iou=report['mean_image_only_iou'],step=step)),flush=True)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f'Received signal {signum}; saving at next optimizer boundary',flush=True)
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum,request_stop)
    model.train()
    if not args.resume: validation()
    progress()
    while not stop_requested and epoch < config['epochs'] and (not config['max_steps'] or step < config['max_steps']):
        generator=torch.Generator().manual_seed(config['seed']+epoch)
        order=[training[i] for i in torch.randperm(len(training),generator=generator).tolist()]
        while not stop_requested and cursor < len(order):
            if config['max_steps'] and step >= config['max_steps']:break
            optimizer.zero_grad(set_to_none=True)
            chunk=order[cursor:cursor+config['gradient_accumulation']]
            log={};started=time.monotonic()
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            for index in chunk:
                sample=to_device(ds[index],device)
                result=rollout(model,sample['image'][None],**config['rollout'])
                terms=trajectory_loss(result,sample,config)
                # Separate backward releases the large point graph before visual
                # supervision, keeping memory bounded to one image/branch graph.
                loss=weighted_loss(terms,config)/len(chunk)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite point loss')
                loss.backward()
                visuals=visual_predictions(model,sample['image'][None],len(sample['stroke_types']))
                vterms=visual_loss(visuals,sample);vloss=weighted_loss(vterms,config)/len(chunk)
                if not torch.isfinite(vloss):raise RuntimeError('nonfinite visual loss')
                vloss.backward()
                for k,v in dict(terms,**vterms).items():log[k]=log.get(k,0.)+float(v.detach())/len(chunk)
                for k,v in dict(generated_points=sum(len(s['geometry']) for s in result['strokes']),
                    predicted_strokes=len(result['strokes']),gt_strokes=len(sample['stroke_types']),
                    completed=int(result['completed']),truncated=int(result['truncated'])).items():
                    log[k]=log.get(k,0.)+v/len(chunk)
                del loss,vloss,terms,vterms,result,visuals
            for group in optimizer.param_groups:
                norm=torch.nn.utils.clip_grad_norm_(group['params'],config['grad_clip'])
                if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
                log[group['name']+'_grad_norm']=float(norm)
            optimizer.step();step+=1;cursor+=len(chunk)
            weights=dict(DEFAULT_WEIGHTS,**config.get('loss_weights',{}))
            log['loss']=sum(weights[k]*log[k] for k in DEFAULT_WEIGHTS)
            log.update(step=step,epoch=epoch,step_seconds=time.monotonic()-started)
            if device.type=='cuda':log['peak_cuda_allocated_mb']=torch.cuda.max_memory_allocated(device)/1024**2
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(json.dumps(log),flush=True)
            if step%config['checkpoint_every']==0:
                save('last.pt');progress()
        if stop_requested:
            save('last.pt');progress('interrupted');ds.close();return
        epoch_complete = cursor==len(order)
        if epoch_complete:epoch+=1;cursor=0
        if epoch_complete and (epoch%config.get('eval_every_epochs',20)==0 or epoch==config['epochs']):
            validation()
            name=f'epoch_{epoch:03d}.pt';save(name)
            gallery=output/f'eval_epoch_{epoch:03d}'
            if gallery.exists():gallery=output/f'eval_epoch_{epoch:03d}_retry_{int(time.time())}'
            try:
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_memorization.py'),
                    '--checkpoint',str(output/name),'--dataset',str(local_path(config['dataset'])),
                    '--device',str(device),'--limit',str(config['validation_limit']),
                    '--max-strokes',str(config['rollout']['max_strokes']),
                    '--max-points',str(config['rollout']['max_points']),
                    '--output',str(gallery)],check=True)
            except subprocess.CalledProcessError:
                if not stop_requested:raise
                save('last.pt');progress('interrupted');ds.close();return
        save('last.pt');progress()
    if stop_requested:
        save('last.pt');progress('interrupted')
    else:
        progress('max_epochs' if epoch>=config['epochs'] else 'max_steps')
    ds.close()


if __name__=='__main__':main()
