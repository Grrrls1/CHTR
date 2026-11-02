"""Convert a character-folder image collection into a resumable, portable trajectory dictionary."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.scripts.infer_image import preprocess_image
from models.scripts.infer_two_stage_batch import atomic_json, sha256
from models.training.ordered_stroke import load_model as load_stroke
from models.training.single_stroke import load_model as load_point, local_path
from models.networks.two_stage import recover_trajectories


def write_preview(path, character, points, offsets, input_name):
    payload = json.dumps(dict(points=points.tolist(), offsets=offsets.tolist()), separators=(',', ':')).replace('<', '\\u003c')
    body = '<!doctype html><meta charset="utf-8"><title>'+html.escape(character)+'</title>'
    body += '<style>body{font:16px sans-serif;margin:24px}img,canvas{width:460px;max-width:100%;border:1px solid #ddd}input{padding:8px}</style>'
    body += '<h1>'+html.escape(character)+' · 柳公权楷书预测轨迹</h1><p>颜色区分预测笔画；数字为笔序，绿圈为起点，红叉为终点，箭头为方向。预测结果未经人工验收。</p>'
    body += '<img src="'+html.escape(input_name, quote=True)+'"><canvas width="768" height="768"></canvas>'
    body += '<script type="application/json" id="data">'+payload+'</script><script>'
    body += '''const d=JSON.parse(document.getElementById('data').textContent),c=document.querySelector('canvas'),x=c.getContext('2d');
x.fillStyle='white';x.fillRect(0,0,768,768);x.scale(3,3);
for(let k=0;k<d.offsets.length-1;k++){
 const p=d.points.slice(d.offsets[k],d.offsets[k+1]);if(!p.length)continue;
 const color=`hsl(${k*137.5%360},75%,38%)`;x.strokeStyle=color;x.fillStyle=color;x.lineWidth=1.5;
 x.beginPath();x.moveTo(p[0][0],p[0][1]);for(const a of p.slice(1))x.lineTo(a[0],a[1]);x.stroke();
 let distance=0;for(let i=1;i<p.length;i++){let dx=p[i][0]-p[i-1][0],dy=p[i][1]-p[i-1][1],n=Math.hypot(dx,dy);distance+=n;
  if(n>0.001&&distance>=22){distance=0;let a=Math.atan2(dy,dx);x.save();x.translate(p[i][0],p[i][1]);x.rotate(a);
   x.strokeStyle='#111';x.lineWidth=1;x.beginPath();x.moveTo(-6,-3);x.lineTo(0,0);x.lineTo(-6,3);x.stroke();x.restore();}}
 const a=p[0],b=p[p.length-1];x.strokeStyle='#15803d';x.lineWidth=1.5;x.beginPath();x.arc(a[0],a[1],3,0,Math.PI*2);x.stroke();
 x.fillStyle=color;x.font='9px sans-serif';x.fillText(String(k+1),a[0]+4,a[1]-4);x.strokeStyle='#dc2626';
 x.beginPath();x.moveTo(b[0]-3,b[1]-3);x.lineTo(b[0]+3,b[1]+3);x.moveTo(b[0]-3,b[1]+3);x.lineTo(b[0]+3,b[1]-3);x.stroke();}
</script>'''
    path.write_text(body, encoding='utf-8')


def write_index(output, index):
    payload = json.dumps(index, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    body = '''<!doctype html><meta charset="utf-8"><title>柳公权楷书轨迹字典</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:24px auto;padding:12px}input,button,select{font:inherit;padding:8px}iframe{width:100%;height:820px;border:1px solid #ddd}</style>
<h1>柳公权楷书轨迹字典</h1><p id="counts"></p><input id="char" maxlength="1" placeholder="输入一个汉字"><button id="search">查询</button><select id="variants"></select><p id="status"></p><iframe title="轨迹预览"></iframe>
<script type="application/json" id="index">'''+payload+'''</script><script>
const d=JSON.parse(document.getElementById('index').textContent),input=document.getElementById('char'),v=document.getElementById('variants'),s=document.getElementById('status'),frame=document.querySelector('iframe');
document.getElementById('counts').textContent=`${d.converted_images}/${d.total_images}幅已处理；${d.available_characters}/${d.total_characters}字可查询。刷新页面读取最新进度。`;
function show(){const e=d.characters[input.value],q=e&&e.variants.find(z=>z.id===v.value);if(!q)return;
 s.textContent=`${input.value} / ${q.id}: ${q.status}。归一化坐标须由demo映射到纸面；每笔之间抬笔。`;
 frame.src=(q.status==='converted'||q.status==='needs_review')?q.preview:'about:blank';}
function search(){v.replaceChildren();const e=d.characters[input.value];if(!e){s.textContent='源数据中没有此字，不进行替换。';frame.src='about:blank';return;}
 for(const q of e.variants){const o=document.createElement('option');o.value=q.id;o.textContent=q.id+' / '+q.status;v.append(o);}
 if(e.default_variant)v.value=e.default_variant;show();}
document.getElementById('search').onclick=search;input.onkeydown=e=>{if(e.key==='Enter')search();};v.onchange=show;input.value=d.characters['你']?'你':Object.keys(d.characters)[0];search();
</script>'''
    temporary = output/'index.html.tmp'
    temporary.write_text(body, encoding='utf-8')
    temporary.replace(output/'index.html')


def build_manifest(source, priority_text='', limit=0):
    extensions = {'.gif', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    rank = {c: i for i, c in enumerate(dict.fromkeys(priority_text))}
    files.sort(key=lambda p: (rank.get(p.parent.name, len(rank)), p.relative_to(source).as_posix()))
    if limit:
        files = files[:limit]
    manifest = []
    seen = set()
    for path in files:
        if len(path.parent.name) != 1 or path.parent.parent != source:
            raise ValueError('Expected source/<single-character>/<image>: '+str(path))
        character, variant = path.parent.name, path.stem
        if (character, variant) in seen:
            raise ValueError('Duplicate character/variant ID: '+str(path))
        seen.add((character, variant))
        with Image.open(path) as image:
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Animated input requires explicit complete-character frame selection: '+str(path))
        manifest.append(dict(character=character, variant_id=variant, source_relative=path.relative_to(source).as_posix(),
                             source_sha256=sha256(path)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/楷-柳公权')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stroke-checkpoint', default='models/checkpoints/ordered_stroke.pt')
    parser.add_argument('--point-checkpoint', default='models/checkpoints/single_stroke.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--point-batch-size', type=int, default=8)
    parser.add_argument('--binarize', choices=['none', 'otsu'], default='none')
    parser.add_argument('--priority-text', default='你好中国柳公权永一人我')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.warmup, args.threads, args.point_batch_size) < 1 or args.limit < 0:
        parser.error('Invalid warmup/threads/batch/limit')
    torch.set_num_threads(args.threads); cv2.setNumThreads(1)
    source, output = local_path(args.source).resolve(), local_path(args.output).resolve()
    if output == source or source in output.parents:
        parser.error('Output must be separate from original source directory')
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output is nonempty; use a new directory or --resume')
    manifest = build_manifest(source, args.priority_text, args.limit)
    if not manifest:
        parser.error('No source images')
    config = dict(schema_version=1, source=str(source), manifest=manifest, device=args.device,
                  preprocessing=dict(resize='fit', margin=8, binarize=args.binarize),
                  checkpoints=[dict(path=str(local_path(p).resolve()), sha256=sha256(local_path(p)))
                               for p in [args.stroke_checkpoint, args.point_checkpoint]],
                  threads=args.threads, point_batch_size=args.point_batch_size, warmup=args.warmup)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'run_config.json').exists():
        if json.loads((output/'run_config.json').read_text()) != config:
            parser.error('Resume source/config/checkpoint differs')
    else:
        atomic_json(output/'run_config.json', config)
    index = dict(schema_version=1, author='柳公权', script_type='楷书', source_directory=str(source),
                 total_images=len(manifest), total_characters=len({m['character'] for m in manifest}),
                 converted_images=0, available_characters=0,
                 coordinates=dict(canvas_size=256, units='pixel XYW; x right, y down; W=image diameter',
                                  normalization_divisors=[255, 255, 256], robot_calibration_applied=False), characters={})
    completed = {}
    for sample in manifest:
        c, v = sample['character'], sample['variant_id']
        stem = f'characters/{c}/{v}'
        item = dict(id=v, status='pending', trajectory_json=stem+'.json', npz=stem+'.npz',
                    input_image=stem+'.input.png', preview=stem+'.html', source_relative=sample['source_relative'])
        report = output/item['trajectory_json']
        if report.exists():
            r = json.loads(report.read_text())
            if r['status'] in ('converted', 'needs_review'):
                item['status'] = r['status']; completed[(c, v)] = r
        index['characters'].setdefault(c, dict(default_variant=None, variants=[]))['variants'].append(item)
    def snapshot():
        states = Counter(v['status'] for e in index['characters'].values() for v in e['variants'])
        for entry in index['characters'].values():
            usable = [v for v in entry['variants'] if v['status'] == 'converted']
            entry['default_variant'] = min((v['id'] for v in usable), default=None)
        index['converted_images'] = states['converted']+states['needs_review']
        index['available_characters'] = sum(e['default_variant'] is not None for e in index['characters'].values())
        atomic_json(output/'dictionary.json', index)
        times = [r['inference_seconds'] for r in completed.values()]
        atomic_json(output/'summary.json', dict(status='complete' if len(completed) == len(manifest) else 'incomplete',
                    total_images=len(manifest), processed_images=len(completed), statuses=dict(states),
                    available_characters=index['available_characters'], total_characters=index['total_characters'],
                    mean_inference_seconds=float(np.mean(times)) if times else None,
                    normalization='XY / 255, diameter / 256',
                    default_selection='smallest source filename among nonempty nontruncated converted variants; no simplified/traditional aliases',
                    note='Model predictions, not human-verified trajectories. No robot coordinates or motion commands.'))
        write_index(output, index)
    snapshot()
    pending = [m for m in manifest if (m['character'], m['variant_id']) not in completed]
    if not pending:
        print('All trajectories already converted', flush=True); return
    stroke, sc = load_stroke(args.stroke_checkpoint, args.device)
    point, pc = load_point(local_path(args.point_checkpoint), args.device)
    stroke.eval(); point.eval()
    device = torch.device(args.device)
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    def input_for(sample):
        image, metadata = preprocess_image(source/sample['source_relative'], binarize=args.binarize)
        return image, metadata, torch.from_numpy(image.astype(np.float32)[None, None]/255.).to(device)
    @torch.inference_mode()
    def infer(inputs):
        return recover_trajectories(stroke, point, inputs,
            max_strokes=sc['config'].get('decode_max_strokes', stroke.max_strokes),
            max_points=pc['config'].get('decode_max_points', point.max_points),
            stroke_threshold=sc['config']['validity_threshold'], point_threshold=pc['config'].get('validity_threshold', .5),
            point_batch_size=args.point_batch_size)[0]
    print(f'{len(manifest)} images, {index["total_characters"]} characters, {len(pending)} pending; warmup {args.warmup}', flush=True)
    for i in range(args.warmup):
        _, _, inputs = input_for(pending[i % len(pending)])
        result = infer(inputs); sync(); del result, inputs
    last_progress = perf_counter()
    for sample in pending:
        c, v = sample['character'], sample['variant_id']
        item = next(z for z in index['characters'][c]['variants'] if z['id'] == v)
        target = output/item['trajectory_json']; target.parent.mkdir(parents=True, exist_ok=True)
        try:
            image, preprocessing, inputs = input_for(sample)
            sync(); start = perf_counter(); result = infer(inputs); sync(); elapsed = perf_counter()-start
            paths = [r['points'].cpu().numpy()*np.array([255, 255, 256], np.float32) for r in result['strokes']]
            points = np.concatenate(paths) if paths else np.empty((0, 3), np.float32)
            offsets = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64)
            if not np.isfinite(points).all() or (len(points) and (points[:, 2] <= 0).any()):
                raise ValueError('Invalid predicted XYW')
            capped = bool(result['instances']['truncated']) or any(r['truncated'] for r in result['strokes'])
            item['status'] = 'needs_review' if capped or not len(paths) else 'converted'
            np.savez_compressed(output/item['npz'], points=points, stroke_offsets=offsets, image_size=256)
            if not cv2.imwrite(str(output/item['input_image']), image): raise OSError('Cannot save input image')
            write_preview(output/item['preview'], c, points, offsets, Path(item['input_image']).name)
            record = dict(schema_version=1, character=c, variant_id=v, status=item['status'], author='柳公权', script_type='楷书',
                          points_xyw_px=points.tolist(), stroke_offsets=offsets.tolist(), stroke_count=len(paths), point_count=len(points),
                          inference_seconds=elapsed, stroke_stop_reason=result['instances']['stop_reason'],
                          point_stop_reasons=[r['stop_reason'] for r in result['strokes']],
                          source_relative=sample['source_relative'], source_sha256=sample['source_sha256'],
                          preprocessing=preprocessing, human_reviewed=False,
                          coordinate_units='256x256 image pixels, image diameter; not robot units or pressure')
            atomic_json(target, record); completed[(c, v)] = record
            del result, inputs
        except Exception as error:
            item['status'] = 'failed'
            atomic_json(target, dict(status='failed', character=c, variant_id=v, error=str(error), traceback=traceback.format_exc()))
            print(f'FAILED {c}/{v}: {error}', flush=True)
            if isinstance(error, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        if len(completed) % 25 == 0 or perf_counter()-last_progress >= 15:
            snapshot(); print(f'{len(completed)}/{len(manifest)} images, {index["available_characters"]}/{index["total_characters"]} characters available', flush=True)
            last_progress = perf_counter()
    snapshot()
    print(json.dumps(dict(images=len(completed), characters=index['available_characters'], output=str(output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
