"""Teacher-forced training; full-split free validation at epoch intervals."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.datasets.single_stroke import collate_single_strokes, SingleStrokeDataset
from models.losses.single_stroke import single_stroke_loss
from models.networks.single_stroke import SingleStrokeModel,MODE
from models.training.single_stroke import local_path,to_device,make_data,signature
from models.training.single_stroke_evaluation import evaluate_split, visualization_ids
from models.training.early_stopping import EarlyStopping


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='models/configs/single_stroke.json')
    p.add_argument('--output-dir');p.add_argument('--device');p.add_argument('--resume')
    p.add_argument('--max-steps',type=int);p.add_argument('--epochs',type=int)
    args=p.parse_args();config=json.loads(local_path(args.config).read_text())
    for k in ('output_dir','device','max_steps','epochs'):
        if getattr(args,k) is not None:config[k]=getattr(args,k)
    if config['training_mode']!=MODE:raise ValueError('incorrect mode')
    if config.get('evaluation_protocol')!='full_split_free_v2':
        raise ValueError('requires full_split_free_v2 config; old runs keep their historical evaluation protocol')
    for k in ('batch_size','epochs','threads','eval_every_epochs','eval_batch_size','checkpoint_every','decode_max_points'):
        if type(config[k]) is not int or config[k]<1:raise ValueError(k+' must be positive')
    for k in ('max_steps','train_char_limit','train_stroke_limit','visualize_characters'):
        if type(config[k]) is not int or config[k]<0:raise ValueError(k+' must be nonnegative')
    if config['decode_max_points']>config['model']['max_points']:raise ValueError('decode cap exceeds model cap')
    if not 0<config['validity_threshold']<1:raise ValueError('invalid validity threshold')
    for k in ('learning_rate','width_weight','valid_weight','grad_clip'):
        if not np.isfinite(config[k]) or config[k]<=0:raise ValueError(k+' must be positive and finite')
    stopper=EarlyStopping(config.get('early_stopping'))
    cosine_steps=config.get('cosine_steps',991000)
    if type(cosine_steps) is not int or cosine_steps<0:raise ValueError('cosine_steps must be nonnegative')
    optimizer_name=config.get('optimizer','adam')
    if optimizer_name not in ('adam','adamw'):raise ValueError('optimizer must be adam or adamw')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config['device']=='auto' else config['device'])
    torch.set_num_threads(config['threads']);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config['seed'])
    output=local_path(config['output_dir']).resolve()
    if args.resume:
        if local_path(args.resume).resolve().parent!=output:raise ValueError('resume must use same output directory')
    elif output.exists() and any(output.iterdir()):raise ValueError('new run needs an empty output directory')
    catalog,train,val,indices,val_indices=make_data(config)
    try:
        sig=signature(config,catalog)
        fixed_ids=visualization_ids(val,config['visualize_characters'])
        model=SingleStrokeModel(**config['model']).to(device)
        optimizer_class=torch.optim.AdamW if optimizer_name=='adamw' else torch.optim.Adam
        optimizer=optimizer_class(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cosine_steps) if cosine_steps
                   else torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.))
        step=epoch=cursor=last_evaluated_epoch=0;best=None;last_val_loss=None
        if args.resume:
            ck=torch.load(local_path(args.resume),map_location=device)
            if ck.get('training_mode')!=MODE or ck['signature']!=sig:raise ValueError('resume architecture/data/split/settings differ; use a new run for changed batch/evaluation protocol')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            step,epoch,cursor=ck['step'],ck['epoch'],ck['cursor']
            best=ck['best_val_free_iou'];last_val_loss=ck['best_val_loss'];last_evaluated_epoch=ck['last_evaluated_epoch']
            if ck['visualization_ids']!=fixed_ids:raise ValueError('fixed visualization IDs changed')
            stopper.load_state_dict(ck['early_stopping'])
            torch.set_rng_state(ck['rng']['torch'].cpu());random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
            if ck['rng']['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        output.mkdir(parents=True,exist_ok=True)
        (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        usage=dict(training_mode=MODE,dataset_fingerprint=catalog.fingerprint,signature=sig,
                   train_characters=len({train.key(i)[0] for i in indices}),train_strokes=len(indices),
                   train_keys=[train.key(i) for i in indices],val_keys=[val.key(i) for i in val_indices],
                   evaluation_split='val',validation_strokes=len(val_indices),visualization_ids=fixed_ids,
                   test_used_for_model_selection=False,evaluation_protocol=config['evaluation_protocol'],
                   training='all true historical XY/width; first XYZ given; subsequent points predicted')
        (output/'data_usage.json').write_text(json.dumps(usage,ensure_ascii=False,indent=2)+'\n')
        (output/'visualization_selection.json').write_text(json.dumps(dict(split='val',sample_ids=fixed_ids),ensure_ascii=False,indent=2)+'\n')
        def save():
            ck=dict(training_mode=MODE,config=config,signature=sig,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,epoch=epoch,cursor=cursor,
                    best_val_free_iou=best,best_val_loss=last_val_loss,last_evaluated_epoch=last_evaluated_epoch,
                    visualization_ids=fixed_ids,rng=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    early_stopping=stopper.state_dict())
            temp=output/'last.pt.tmp';torch.save(ck,temp);temp.replace(output/'last.pt')
        def fresh_evaluation_path(path):
            if path.exists():path.rename(path.with_name(path.name+f'_previous_{time.time_ns()}'))
            return path
        def validate(log):
            nonlocal best,last_evaluated_epoch,last_val_loss
            target=fresh_evaluation_path(output/f'eval_total_epoch_{epoch:03d}')
            report=evaluate_split(model,val,device,config,target,fixed_ids,val_indices)
            score=report['free_trajectory_iou'];last_val_loss=report['teacher_forced']['loss']
            improved=best is None or score>best
            if improved:best=score
            # Patience counts validation rounds (10 epochs per round in the full config).
            stopper.add(1.-score,1);_,significant,stop=stopper.finish_epoch(epoch)
            last_evaluated_epoch=epoch
            report.update(epoch=epoch,step=step,selection_metric='free_trajectory_iou',best_val_free_iou=best)
            (target/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            record=dict(kind='validation',step=step,epoch=epoch,split='val',strokes=report['strokes'],
                        full_split=report['full_split'],teacher_forced=report['teacher_forced'],free=report['free'],
                        selection_metric='free_trajectory_iou',best_val_free_iou=best,gallery=str(target/'index.html'))
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            record=dict(kind='early_stopping',step=step,epoch=epoch,monitor='1-free_trajectory_iou',
                        bad_evaluations=stopper.bad_epochs,improved=significant,stop=stop)
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            save()
            if improved:shutil.copyfile(output/'last.pt',output/'best.pt')
        start=time.monotonic();model.train();order_epoch=None;order=None
        with (output/'metrics.jsonl').open('a') as log:
            while epoch<config['epochs'] and not stopper.should_stop and (not config['max_steps'] or step<config['max_steps']):
                if order_epoch!=epoch:
                    order=torch.randperm(len(indices),generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()
                    order_epoch=epoch
                batch_indices=order[cursor:cursor+config['batch_size']]
                samples=[train[indices[i]] for i in batch_indices]
                batch=to_device(collate_single_strokes(samples),device)
                optimizer.zero_grad(set_to_none=True)
                predictions=model(batch['images'],batch['history'],batch['active'])
                loss,terms=single_stroke_loss(predictions,batch,config['width_weight'],config['valid_weight'])
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
                if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
                optimizer.step();scheduler.step();step+=1;cursor+=len(samples)
                if cursor==len(indices):epoch+=1;cursor=0
                row=dict(kind='train',step=step,epoch=epoch,cursor=cursor,grad_norm=float(norm),
                         **{k:float(v.detach()) for k,v in terms.items()},
                         peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if cursor==0 and epoch%config['eval_every_epochs']==0:validate(log)
                elif step%config['checkpoint_every']==0 or cursor==0:save()
            # A step limit does not trigger an extra, partial-epoch validation.
            save()
        reason=('early_stopping' if stopper.should_stop else
                'max_steps' if config['max_steps'] and step>=config['max_steps'] else 'epochs')
        status=dict(step=step,epoch=epoch,cursor=cursor,device=str(device),seconds=time.monotonic()-start,
                    best_val_free_iou=best,last_val_teacher_forced_loss=last_val_loss,last_evaluated_epoch=last_evaluated_epoch,
                    parameters=sum(p.numel() for p in model.parameters()),reason=reason,
                    early_stopping_monitor='1-free_trajectory_iou',early_stopping=stopper.state_dict())
        (output/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if config.get('final_test',False) and reason in ('epochs','early_stopping'):
            if not (output/'best.pt').exists():raise ValueError('no validated best checkpoint; final test requires at least one scheduled validation')
            test_output=output/'test_best'
            ck=torch.load(output/'best.pt',map_location=device)
            existing=json.loads((test_output/'summary.json').read_text()) if (test_output/'summary.json').exists() else {}
            if existing.get('checkpoint_step')!=ck['step']:
                model.load_state_dict(ck['model'])
                test=SingleStrokeDataset(catalog,'test',config['model']['max_points'])
                report=evaluate_split(model,test,device,config,fresh_evaluation_path(test_output),visualization_ids(test,config['visualize_characters']))
                report.update(checkpoint=str(output/'best.pt'),checkpoint_epoch=ck['epoch'],checkpoint_step=ck['step'],test_used_for_model_selection=False)
                (test_output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    finally:catalog.close()


if __name__=='__main__':main()
