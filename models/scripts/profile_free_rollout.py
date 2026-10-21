"""Profile real image-only rollout backward, without updating any model weights."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from models.datasets.lmdb_dataset import CHTRModelDataset
from models.training.free_rollout_setup import create_model
from models.networks.free_rollout import rollout,visual_predictions
from models.losses.free_rollout_loss import trajectory_loss,visual_loss,weighted_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/free_rollout.json'))
    p.add_argument('--sample-ids',nargs='+',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True)
    p.add_argument('--checkpoint', help='Explicit matching checkpoint; omitted means random initialization')
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    if output.exists():raise ValueError('use a new profiling output file')
    def resolve(value):
        path=Path(value);return path if path.is_absolute() else ROOT/path
    torch.set_num_threads(config.get('threads',2));device=torch.device(args.device)
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config['seed'])
    m,_=create_model(config,device,resolve(args.checkpoint) if args.checkpoint else None)
    m.eval()
    ds=CHTRModelDataset(resolve(config['dataset']));indices={sid:i for i,sid in enumerate(ds.ids)}
    reports=[]
    for sid in args.sample_ids:
        s={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in ds[indices[sid]].items()}
        m.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        start=time.monotonic();r=rollout(m,s['image'][None],**config['rollout'])
        terms=trajectory_loss(r,s,config);loss=weighted_loss(terms,config);loss.backward()
        visual=visual_predictions(m,s['image'][None],len(s['stroke_types']))
        vt=visual_loss(visual,s);vl=weighted_loss(vt,config);vl.backward()
        if device.type=='cuda':torch.cuda.synchronize(device)
        row=dict(sample_id=sid,character=s['character'],gt_strokes=len(s['stroke_types']),
            generated_strokes=len(r['strokes']),generated_points=sum(len(v['geometry']) for v in r['strokes']),
            completed=r['completed'],truncation_reason=r['truncation_reason'],seconds=time.monotonic()-start,
            finite_gradients=all(bool(torch.isfinite(v.grad).all()) for v in m.parameters() if v.grad is not None),
            terms={k:float(v.detach()) for k,v in dict(terms,**vt).items()})
        if device.type=='cuda':row['peak_cuda_allocated_mb']=torch.cuda.max_memory_allocated(device)/1024**2
        reports.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        del r,terms,loss,visual,vt,vl
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n');ds.close()


if __name__=='__main__':main()
