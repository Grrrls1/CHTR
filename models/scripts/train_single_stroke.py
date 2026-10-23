"""Independent teacher-forced XY/width/validity training. No Slurm generation."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.datasets.single_stroke import collate_single_strokes
from models.losses.single_stroke import single_stroke_loss
from models.networks.single_stroke import SingleStrokeModel,MODE
from models.training.single_stroke import local_path,to_device,make_data,signature,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='models/configs/single_stroke.json')
    p.add_argument('--output-dir');p.add_argument('--device');p.add_argument('--resume')
    p.add_argument('--max-steps',type=int);p.add_argument('--epochs',type=int)
    args=p.parse_args();config=json.loads(local_path(args.config).read_text())
    for k in ('output_dir','device','max_steps','epochs'):
        if getattr(args,k) is not None:config[k]=getattr(args,k)
    if config['training_mode']!=MODE:raise ValueError('incorrect mode')
    for k in ('batch_size','epochs','eval_strokes','eval_every_steps','threads'):
        if type(config[k]) is not int or config[k]<1:raise ValueError(k+' must be positive')
    for k in ('max_steps','train_char_limit','train_stroke_limit'):
        if type(config[k]) is not int or config[k]<0:raise ValueError(k+' must be nonnegative')
    for k in ('learning_rate','width_weight','valid_weight','grad_clip'):
        if not np.isfinite(config[k]) or config[k]<=0:raise ValueError(k+' must be positive and finite')
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
        model=SingleStrokeModel(**config['model']).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.get('cosine_steps',991000))
        step=epoch=cursor=0;best=float('inf')
        if args.resume:
            ck=torch.load(local_path(args.resume),map_location=device)
            if ck.get('training_mode')!=MODE or ck['signature']!=sig:raise ValueError('resume architecture/data/split/settings differ')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            step,epoch,cursor,best=ck['step'],ck['epoch'],ck['cursor'],ck['best_val_loss']
            torch.set_rng_state(ck['rng']['torch'].cpu());random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
            if ck['rng']['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        output.mkdir(parents=True,exist_ok=True)
        (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        usage=dict(training_mode=MODE,dataset_fingerprint=catalog.fingerprint,signature=sig,
                   train_characters=len({train.key(i)[0] for i in indices}),train_strokes=len(indices),
                   train_keys=[train.key(i) for i in indices],val_keys=[val.key(i) for i in val_indices],
                   evaluation_split='val',test_used_for_model_selection=False,
                   training='all true historical XY/width; first XYZ given; subsequent points predicted')
        (output/'data_usage.json').write_text(json.dumps(usage,ensure_ascii=False,indent=2)+'\n')
        def save():
            ck=dict(training_mode=MODE,config=config,signature=sig,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,epoch=epoch,cursor=cursor,
                    best_val_loss=best,rng=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
            temp=output/'last.pt.tmp';torch.save(ck,temp);temp.replace(output/'last.pt')
        def validate(log):
            nonlocal best
            metrics=evaluate(model,val,val_indices,device,config)
            record=dict(kind='validation',step=step,split='val',**metrics)
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            improved=metrics['loss']<best
            if improved:best=metrics['loss']
            save()
            if improved:
                import shutil
                shutil.copyfile(output/'last.pt',output/'best.pt')
            return metrics
        start=time.monotonic();model.train();last_eval_step=-1
        with (output/'metrics.jsonl').open('a') as log:
            while epoch<config['epochs'] and (not config['max_steps'] or step<config['max_steps']):
                order=torch.randperm(len(indices),generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()
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
                         **{k:float(v.detach()) for k,v in terms.items()})
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if step%config['eval_every_steps']==0:
                    validate(log);last_eval_step=step
            if step!=last_eval_step:validation=validate(log)
            save()
        status=dict(step=step,epoch=epoch,cursor=cursor,device=str(device),seconds=time.monotonic()-start,
                    best_val_loss=best,parameters=sum(p.numel() for p in model.parameters()),
                    reason='max_steps' if config['max_steps'] and step>=config['max_steps'] else 'epochs')
        (output/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
    finally:catalog.close()


if __name__=='__main__':main()
