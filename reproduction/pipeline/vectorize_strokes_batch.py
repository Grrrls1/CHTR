"""Resident TF1 model, deterministic candidate seeds, content-checked resume."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import sys
import numpy as np
from common import ROOT, records, unique, identifier, require, gray, digest, write_json, read_json, write_records, arrays


DEFAULT_ROUNDS = 10
DEFAULT_MAX_STEPS = 500
DEFAULT_ROUND_STOP_STATE_NUM = 12
DEFAULT_STROKE_ACC_THRESHOLD = .95


def _skeleton_endpoints(mask):
    """Return endpoint pixels as ``(x, y)`` pairs without importing TensorFlow."""
    import cv2
    mask = np.asarray(mask, dtype=np.uint8)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        skeleton = cv2.ximgproc.thinning(mask * 255) > 0
    else:
        try:
            from skeleton_path_trajectory import skeletonize_mask
            skeleton = skeletonize_mask(mask)
        except ImportError:
            skeleton = mask.astype(bool)
    ys, xs = np.where(skeleton)
    endpoints = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - 1), min(mask.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(mask.shape[1], x + 2)
        if int(skeleton[y0:y1, x0:x1].sum()) == 2:
            endpoints.append((x, y))
    return np.asarray(endpoints, dtype=np.float32)


def _median_path_for_row(row, mmah_by_character):
    """Load the optional MMAH median for one stroke, or return ``None``."""
    import os
    source = row.get('mmah_source_npz_path')
    if not source:
        reference = mmah_by_character.get(row.get('character'))
        source = reference.get('source_npz_path') if reference else None
    if not source or not os.path.isfile(source):
        return None
    with np.load(source, allow_pickle=False) as z:
        key = 'median_%03d' % int(row.get('stroke_index', 0))
        if key not in z:
            return None
        median = np.asarray(z[key], dtype=np.float32)
    if median.ndim != 2 or median.shape[0] < 2 or median.shape[1] != 2 or not np.isfinite(median).all():
        return None
    return median


def resolve_start_cursor(source, initial_cursor, row, mmah_by_character=None):
    """Resolve the first cursor in normalized ``[x, y]`` canvas coordinates."""
    require(source in ('random', 'skeleton', 'mmah'), 'Unknown start cursor source: ' + str(source))
    initial = np.asarray(initial_cursor, dtype=np.float32)
    require(initial.ndim >= 2 and initial.shape[-2:] == (1, 2),
            'Unexpected loader cursor shape: ' + str(initial.shape))
    if source == 'random':
        return initial, dict(source='random', status='loader_random')
    mask = (gray(row['input_png']) < 128).astype(np.uint8)
    endpoints = _skeleton_endpoints(mask)
    require(len(endpoints) > 0, 'No skeleton endpoint for start cursor: ' + row['stroke_id'])
    selected = endpoints[np.lexsort((endpoints[:, 0], endpoints[:, 1]))[0]]
    detail = dict(source=source, status='skeleton_endpoint', endpoint_count=int(len(endpoints)))
    if source == 'mmah':
        median = _median_path_for_row(row, mmah_by_character or {})
        require(median is not None, 'MMAH median unavailable for: ' + row['stroke_id'])
        distances = np.linalg.norm(endpoints - median[0][None, :], axis=1)
        selected = endpoints[int(np.argmin(distances))]
        detail.update(status='mmah_median_endpoint', median_start=median[0].tolist(),
                      median_end=median[-1].tolist(), endpoint_distance=float(np.min(distances)))
    size = float(mask.shape[0])
    cursor = np.clip(selected / size, 0.0, np.nextafter(1.0, 0.0)).astype(np.float32)
    return cursor.reshape(initial.shape), detail


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--repo',default=str(ROOT/'CHTR/third_party/virtual_sketching')); p.add_argument('--model',default='pretrain_clean_line_drawings')
    p.add_argument('--samples',type=int,default=3); p.add_argument('--seed',type=int,default=0); p.add_argument('--resume',action='store_true')
    p.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS)
    p.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument('--round-stop-state-num', type=int, default=DEFAULT_ROUND_STOP_STATE_NUM)
    p.add_argument('--stroke-acc-threshold', type=float, default=DEFAULT_STROKE_ACC_THRESHOLD)
    p.add_argument('--min-width', type=float, default=None)
    p.add_argument('--min-window-size', type=float, default=None)
    p.add_argument('--start-cursor-source', choices=('random', 'skeleton', 'mmah'), default='random')
    p.add_argument('--vs-device', choices=('cpu', 'cuda'), default='cpu',
                   help='Virtual Sketching model device; default preserves historical CPU inference')
    p.add_argument('--mmah-manifest', default=None,
                   help='Optional MMAH ordered-NPZ manifest for --start-cursor-source mmah')
    a=p.parse_args()
    require(a.samples>0 and a.rounds>0 and a.max_steps>0 and a.round_stop_state_num>0,
            'samples, rounds, max-steps and round-stop-state-num must be positive')
    require(0 < a.stroke_acc_threshold <= 1, 'stroke-acc-threshold must be in (0,1]')
    if a.min_width is not None:
        require(0 < a.min_width <= 1, 'min-width must be in (0,1]')
    if a.min_window_size is not None:
        require(0 < a.min_window_size <= 256, 'min-window-size must be in (0,256]')
    if a.start_cursor_source == 'mmah':
        require(a.mmah_manifest, '--mmah-manifest is required for --start-cursor-source mmah')
    identifier(a.model)
    rows=records(a.manifest); unique(rows,'stroke_id'); out=Path(a.output).resolve()
    require(a.resume or not out.exists() or not any(out.iterdir()),'Output exists; use --resume with identical inputs/config')
    out.mkdir(parents=True,exist_ok=True)
    repo=Path(a.repo).resolve(); model_dir=repo/'outputs/snapshot'/a.model
    config=read_json(model_dir/'model_config.json')
    effective_config=dict(config)
    if a.min_width is not None:
        effective_config['min_width']=a.min_width
    if a.min_window_size is not None:
        effective_config['min_window_size']=a.min_window_size
    inference=dict(rounds=a.rounds, max_steps=a.max_steps,
                   round_stop_state_num=a.round_stop_state_num,
                   stroke_acc_threshold=a.stroke_acc_threshold,
                   min_width=a.min_width if a.min_width is not None else config.get('min_width'),
                   min_window_size=a.min_window_size if a.min_window_size is not None else config.get('min_window_size'),
                   start_cursor_source=a.start_cursor_source, vs_device=a.vs_device)
    mmah_by_character={}
    if a.mmah_manifest:
        mmah_rows=records(a.mmah_manifest)
        mmah_by_character={r['character']: r for r in mmah_rows}
        require(len(mmah_by_character) == len(mmah_rows), 'Duplicate MMAH character')
    model_files=sorted(model_dir.glob('*.index'))+sorted(model_dir.glob('*.data-*'))
    require(model_files,'Missing checkpoint')
    model_hash={v.name:digest(v) for v in model_files}
    code_hash={str(v.relative_to(repo)):digest(v) for v in sorted(repo.glob('*.py'))}
    code_hash['tools/visualize_drawing.py']=digest(repo/'tools/visualize_drawing.py')
    code_hash['pipeline_worker']=digest(Path(__file__))
    os.environ.update(PYTHONNOUSERSITE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
    sys.path.insert(0,str(repo)); os.chdir(str(repo))
    import tensorflow as tf
    import hyper_parameters as hp
    from utils import update_hyperparams, reset_graph, load_checkpoint, save_seq_data
    from dataset_utils import load_dataset_testing
    from test_vectorization import sample
    from model_common_test import VirtualSketchingModel, DiffPastingV3
    from tools.visualize_drawing import display_strokes_final
    require(tf.__version__.startswith('1.'),'Run with isolated TF1 environment')
    params=update_hyperparams(hp.get_default_hparams_clean(),str(model_dir.parent),a.model,infer_dataset='clean_line_drawings')
    if a.min_width is not None:
        params.min_width=a.min_width
    if a.min_window_size is not None:
        params.min_window_size=a.min_window_size
    require(not params.init_cursor_on_undrawn_pixel,'Only round-reset width model supported')
    require(rows,'Empty manifest')
    _,eval_hps,sample_hps=load_dataset_testing('', 'clean_line_drawings',rows[0]['input_png'],params)
    reset_graph(); tf.set_random_seed(a.seed)
    model=VirtualSketchingModel(sample_hps,gpu_mode=(a.vs_device == 'cuda')); paste=DiffPastingV3(sample_hps.raster_size)
    session_config=tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
    if a.vs_device == 'cuda':
        session_config.gpu_options.allow_growth=True
    sess=tf.InteractiveSession(config=session_config)
    sess.run(tf.global_variables_initializer()); step=load_checkpoint(sess,str(model_dir),gen_model_pretrain=True)
    write_json(out/'model_config.json',effective_config); result=[]; errors=[]
    mmah_manifest_sha256 = digest(a.mmah_manifest) if a.mmah_manifest else None
    completed_candidates = 0
    for r in rows:
        for candidate in range(a.samples):
            cid=r['stroke_id']+'_c%03d'%candidate
            try:
                im=gray(r['input_png']); require(im.shape==(256,256) and np.any(im<128),'Expected nonempty 256 PNG')
                seed=int(hashlib.sha256((str(a.seed)+cid).encode()).hexdigest()[:8],16)
                mmah_source_sha256=None
                if a.start_cursor_source == 'mmah':
                    reference = mmah_by_character.get(r.get('character'))
                    source = r.get('mmah_source_npz_path') or (reference.get('source_npz_path') if reference else None)
                    require(source and os.path.isfile(source), 'MMAH source missing for: ' + r['stroke_id'])
                    mmah_source_sha256=digest(source)
                signature=dict(input_sha256=digest(r['input_png']),model_sha256=model_hash,code_sha256=code_hash,
                              mmah_manifest_sha256=mmah_manifest_sha256,
                              mmah_source_sha256=mmah_source_sha256,
                              config=effective_config,inference=inference,seed=seed,step=int(step))
                d=out/cid; meta=d/'candidate.json'
                if meta.exists():
                    old=read_json(meta); require(old['signature']==signature,'Resume fingerprint changed: '+cid)
                    require(digest(old['npz'])==old['npz_sha256'],'Cached NPZ changed')
                    require(digest(old['upstream_render_path'])==old['upstream_render_sha256'],'Cached render changed')
                    result.append(dict(r,**old)); continue
                d.mkdir(parents=True,exist_ok=True)
                random.seed(seed); np.random.seed(seed)
                loader,_,_=load_dataset_testing('', 'clean_line_drawings',r['input_png'],params)
                photos,init,size=loader.get_test_image(); init=init[None]
                init, cursor_detail = resolve_start_cursor(a.start_cursor_source, init, r, mmah_by_character)
                sampled=sample(sess,model,photos,init,size,eval_hps.max_seq_len,
                               [a.max_steps]*a.rounds,False,paste,
                               a.round_stop_state_num,a.stroke_acc_threshold)
                pars,_,_,_,_,new_cursors,lengths=sampled
                data=np.stack(pars[0]); cursors=np.asarray([init[0,0,0]]+[c[0,0] for c in new_cursors])
                save_seq_data(str(d),'candidate',data,cursors,size,lengths,eval_hps.min_width)
                npz=d/'seq_data/candidate.npz'; vis=d/'visualization'; vis.mkdir(exist_ok=True)
                display_strokes_final(sess,paste,data,cursors,size,lengths,eval_hps.min_width,str(vis),min_window_size=sample_hps.min_window_size,raster_size=sample_hps.raster_size)
                render=vis/'output_rendered.png'
                item=dict(candidate_id=cid,candidate_index=candidate,npz=str(npz),npz_sha256=digest(npz),
                          upstream_render_path=str(render),upstream_render_sha256=digest(render),
                          model_config=str(out/'model_config.json'),inference=inference,
                          start_cursor=cursor_detail,signature=signature)
                write_json(meta,item); result.append(dict(r,**item)); print('Completed '+cid,flush=True)
            except Exception as e:
                errors.append(dict(candidate_id=cid,error=str(e))); print('Failed '+cid+': '+str(e),flush=True)
            completed_candidates += 1
            # Each candidate already has atomic metadata for resume. Avoid
            # rewriting the growing full manifest on every single candidate.
            if completed_candidates % 100 == 0:
                write_records(out/'manifest.jsonl',result); write_records(out/'errors.jsonl',errors)
    sess.close(); write_records(out/'manifest.jsonl',result); write_records(out/'errors.jsonl',errors)
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
