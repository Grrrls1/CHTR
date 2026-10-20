"""Add predicted type-colored masks to existing galleries without changing trajectories."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from models.datasets.lmdb_dataset import CHTRModelDataset
from models.networks.model import StrokeTrajectoryModel
from models.networks.autoregressive import encode_image,next_stroke
from models.visualization.mask_types import save_masks,write_gallery


def add_npz_fields(path,fields):
    with np.load(path,allow_pickle=False) as z:old={key:z[key] for key in z.files}
    old.update(fields)
    temporary=path.with_name(path.stem+'.with_masks.tmp.npz')
    np.savez_compressed(temporary,**old);temporary.replace(path)


def export(folder,device):
    folder=Path(folder).resolve();summary=json.loads((folder/'summary.json').read_text())
    checkpoint_path=Path(summary['checkpoint'])
    with (folder/'.mask_types.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        config=checkpoint['config'];model=StrokeTrajectoryModel(**config['model']).to(device)
        model.load_state_dict(checkpoint['model']);model.eval()
        dataset=CHTRModelDataset(config['dataset']);lookup={sid:i for i,sid in enumerate(dataset.ids)}
        backup=folder/'index.before_mask_types.html'
        if (folder/'index.html').exists() and not backup.exists():shutil.copyfile(folder/'index.html',backup)
        sheet_backup=folder/'comparison_sheet.before_mask_types.png'
        if (folder/'comparison_sheet.png').exists() and not sheet_backup.exists():
            shutil.copyfile(folder/'comparison_sheet.png',sheet_backup)
        with torch.inference_mode():
            for index,result in enumerate(summary['results']):
                sid=result['sample_id'];sample=dataset[lookup[sid]];directory=folder/sid
                with np.load(directory/'prediction.npz',allow_pickle=False) as z:
                    original_types=z['stroke_types'].copy();free_count=len(original_types)
                    if len(z['stroke_offsets'])!=free_count+1:raise ValueError('Free stroke/type count mismatch')
                teacher_count=len(sample['stroke_types']);count=max(teacher_count,free_count)
                if not 1<=count<=model.max_strokes:raise ValueError('Invalid saved stroke count')
                image=sample['image'][None].to(device);pixels,memory,sizes=encode_image(model,image)
                history=[];masks=[];probabilities=[]
                for _ in range(count):
                    feature,mask,types,_,_=next_stroke(model,pixels,memory,sizes,history,sample['image_size'])
                    history.append(feature);masks.append(mask.sigmoid()[0]);probabilities.append(types.softmax(-1)[0])
                masks=torch.stack(masks).cpu().numpy();probabilities=torch.stack(probabilities).cpu().numpy()
                if not np.array_equal(probabilities[:free_count].argmax(-1),original_types):
                    raise ValueError('Recomputed free types differ from saved generation: '+sid)
                save_masks(directory,np.rint(sample['image'][0].numpy()*255).astype(np.uint8),dataset.names,
                           sample['stroke_masks'].numpy(),sample['stroke_types'].numpy(),
                           masks[:teacher_count],probabilities[:teacher_count],masks[:free_count],probabilities[:free_count])
                add_npz_fields(directory/'teacher_forced.npz',dict(stroke_masks=masks[:teacher_count],
                    stroke_types=probabilities[:teacher_count].argmax(-1),stroke_type_probs=probabilities[:teacher_count]))
                add_npz_fields(directory/'prediction.npz',dict(stroke_masks=masks[:free_count],stroke_type_probs=probabilities[:free_count]))
                write_gallery(folder,summary['results'],dataset.names)
                print('[mask types] %d/%d %s GT_slots=%d free_slots=%d'%(index+1,len(summary['results']),sid,teacher_count,free_count),flush=True)
        write_gallery(folder,summary['results'],dataset.names)
        metadata=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      images=len(summary['results']),mask_threshold=.5,color_by='stroke_type_id',
                      free_count_source='original image-only generated stroke count',
                      predicted_type_source='checkpoint type-head argmax',
                      existing_trajectories_and_metrics_unchanged=True)
        (folder/'mask_type_summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
        dataset.close();print('[mask types done] '+str(folder/'index.html'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True,action='append');p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.set_num_threads(2)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('Slurm GPU allocation required')
    for folder in args.eval_dir:export(folder,args.device)


if __name__=='__main__':main()
