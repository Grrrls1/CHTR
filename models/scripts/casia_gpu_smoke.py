"""Bounded real-POT two-stage integration check. Run only inside a Slurm GPU job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.device != 'cpu' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('GPU smoke must run through gpu-sbatch --user_name lkh')
    began=time.monotonic()
    if a.device != 'cpu' and not torch.cuda.is_available():raise RuntimeError('allocated CUDA unavailable')
    def run(stage,config,extra=()):
        path=out/(stage+'_config.json');path.write_text(json.dumps(config,indent=2)+'\n')
        subprocess.run([sys.executable,'-u',f'models/scripts/train_{stage}.py','--config',str(path),'--device',a.device,*extra],check=True)
    point=json.loads(Path('models/configs/casia_single_stroke.json').read_text())
    point.update(output_dir=str(out/'point'),batch_size=2,eval_batch_size=2,epochs=2,max_steps=2,
                 train_char_limit=2,train_stroke_limit=4,eval_char_limit=2,visualize_characters=2,
                 decode_max_points=8,checkpoint_every=1,final_test=False)
    run('single_stroke',point)
    run('single_stroke',point,['--resume',str(out/'point/last.pt'),'--max-steps','3'])
    stroke=json.loads(Path('models/configs/casia_ordered_stroke.json').read_text())
    stroke.update(output_dir=str(out/'stroke'),point_checkpoint=str(out/'point/best.pt'),
                  point_decode_max_points=8,point_batch_size=2,batch_size=1,eval_batch_size=1,epochs=2,
                  max_steps=2,train_char_limit=2,eval_char_limit=2,decode_max_strokes=2,
                  checkpoint_every=1,visualize_characters=2)
    run('ordered_stroke',stroke)
    run('ordered_stroke',stroke,['--resume',str(out/'stroke/last.pt'),'--max-steps','3'])
    # A separate evaluation reloads the trained pair and validates signature/provenance.
    subprocess.run([sys.executable,'-u','models/scripts/evaluate_ordered_stroke.py',
                    '--checkpoint',str(out/'stroke/best.pt'),'--output',str(out/'independent_val'),
                    '--split','val','--limit','2','--device',a.device],check=True)
    point_status=json.loads((out/'point/training_status.json').read_text())
    stroke_status=json.loads((out/'stroke/training_status.json').read_text())
    report=dict(status='passed',job_id=os.environ.get('SLURM_JOB_ID'),user=os.environ.get('USER'),
                device=torch.cuda.get_device_name(0) if a.device != 'cpu' else 'cpu',seconds=time.monotonic()-began,
                point=point_status,stroke=stroke_status,
                scope='real train/val POT; 3 optimizer steps each incl resume; full Swin-T/9 layers; bounded decoding; no test and no quality/capacity acceptance')
    (out/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
