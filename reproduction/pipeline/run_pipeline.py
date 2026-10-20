"""Run explicit JSON-configured stages with separate interpreters and checked resume."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import read_json, write_json, digest, require, identifier

STAGES={'prepare_sources','build_reference_bank','extract_strokes_batch','clean_stroke_masks','review_strokes','export_vs_inputs','export_clean_masks_for_vs','vectorize_strokes_batch','render_and_filter','review_trajectories','assemble_vs_exploratory','pack_dataset','validate_dataset','decode_vs_npz','select_single_path_eval','skeleton_path_trajectory','assemble_single_path','compare_single_path_routes','render_character_centerlines','pack_exploratory_example','simplify_dataset_manifest'}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); config_path=Path(a.config).resolve(); cfg=read_json(config_path)
    os.chdir(str(config_path.parent))
    report=Path(cfg['report']).resolve(); old=read_json(report) if report.exists() and a.resume else {}; results={}
    env=dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
    require(len({s['id'] for s in cfg['stages']})==len(cfg['stages']),'Duplicate stage ID')
    for stage in cfg['stages']:
        identifier(stage['id'])
        name=stage['script']; require(name in STAGES,'Unknown stage: '+name)
        script=Path(__file__).parent/(name+'.py'); cmd=[stage.get('python',sys.executable),str(script)]+stage['args']
        print('Stage '+stage['id']+': '+repr(cmd),flush=True)
        if a.dry_run: continue
        require(stage.get('inputs') and stage.get('outputs'),'Each stage requires explicit input/output files for resume')
        signature=dict(command=cmd,script=digest(script),inputs={v:digest(v) for v in stage['inputs']},code={str(v.name):digest(v) for v in script.parent.glob('*.py')})
        cached=old.get(stage['id'])
        if cached and cached.get('returncode')==0 and cached['signature']==signature and all(Path(v).is_file() and digest(v)==h for v,h in cached['outputs'].items()):
            results[stage['id']]=cached; continue
        if stage.get('requires_decisions'):
            require(Path(stage['requires_decisions']).is_file(),'Human decisions file required: '+stage['requires_decisions'])
        log=report.parent/(stage['id']+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f: proc=subprocess.run(cmd,cwd=str(config_path.parent),env=env,stdout=f,stderr=subprocess.STDOUT)
        result=dict(signature=signature,returncode=proc.returncode,log=str(log),outputs={v:digest(v) for v in stage['outputs'] if Path(v).is_file()})
        if proc.returncode==0 and len(result['outputs'])!=len(stage['outputs']): result['returncode']=1
        results[stage['id']]=result; write_json(report,results)
        if result['returncode']:
            raise SystemExit('Stage failed or waiting for review; see '+str(log))
    if not a.dry_run: write_json(report,results)


if __name__=='__main__': main()
