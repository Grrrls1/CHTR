"""Direction-aware trajectory galleries from pixel XYW, with pen-up boundaries."""
import html
import base64
import json
from pathlib import Path

import cv2
import numpy as np


PALETTE = ('#2563eb', '#9333ea', '#d97706', '#0891b2', '#be185d', '#4d7c0f')
STYLE = '''<style>
body{font:15px system-ui,sans-serif;max-width:1240px;margin:24px auto;padding:0 16px;color:#172033;background:#f6f8fc}
section,article{background:white;border:1px solid #dce2ec;border-radius:12px;padding:16px;margin:18px 0}
.panels{display:flex;gap:20px;flex-wrap:wrap}.panel{flex:1;min-width:260px;max-width:560px}
.panel svg{display:block;width:100%;height:auto;border:1px solid #e2e8f0;background:white}
.coverage img{width:256px;max-width:100%;height:auto}.coverage figure{margin:0}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 0}.controls input{flex:1;min-width:180px}
button,select{font:inherit;padding:5px 12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.note{color:#536176}.start{color:#15803d}.end{color:#dc2626}a{color:#1d4ed8}
</style>'''
SCRIPT = '''<script>
document.querySelectorAll('[data-player]').forEach(player=>{
 const slider=player.querySelector('input[type=range]'),button=player.querySelector('button'),
 speed=player.querySelector('select'),status=player.querySelector('output');
 let timer=null;
 function draw(){const f=Number(slider.value)/1000;
  player.querySelectorAll('svg[data-total]').forEach(svg=>{
   const n=Math.floor(Number(svg.dataset.total)*f);
   svg.querySelectorAll('[data-step]').forEach(el=>{el.style.display=Number(el.dataset.step)<=n?'':'none';});
  });status.textContent=(f*100).toFixed(1)+'%';
 }
 function stop(){clearInterval(timer);timer=null;button.textContent='播放';}
 slider.addEventListener('input',()=>{stop();draw();});
 button.addEventListener('click',()=>{
  if(timer){stop();return;}
  if(Number(slider.value)>=1000)slider.value=0;
  button.textContent='暂停';draw();
  timer=setInterval(()=>{slider.value=Math.min(1000,Number(slider.value)+5*Number(speed.value));draw();
   if(Number(slider.value)>=1000)stop();},50);
 });draw();
});
</script>'''


def checked_paths(paths):
    result = []
    for path in paths:
        path = np.asarray(path, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 3 or not len(path) or not np.isfinite(path).all() or (path[:, 2] <= 0).any():
            raise ValueError('expected nonempty finite Nx3 pixel XYW with positive width')
        result.append(path)
    return result


def trajectory_svg(paths, labels=None):
    """Each path is an independent pen-down stroke; never draw between paths."""
    paths = checked_paths(paths)
    labels = list(range(1, len(paths)+1)) if labels is None else list(labels)
    if len(labels) != len(paths):
        raise ValueError('one label required per stroke')
    total = sum(map(len, paths))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-12 -12 280 280" data-total="{total}" role="img" aria-label="Ordered XYW trajectory">',
             '<rect x="0" y="0" width="255" height="255" fill="white" stroke="#cbd5e1"/>']
    offset = 0
    for k, (path, label) in enumerate(zip(paths, labels)):
        color = PALETTE[k % len(PALETTE)]
        xy = path[:, :2]
        for i in range(1, len(path)):
            a, b = xy[i-1], xy[i]
            parts.append(f'<line data-step="{offset+i+1}" x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" y2="{b[1]:.3f}" stroke="{color}" stroke-width="1.6"/>')
        # Place arrows along travelled distance, ignoring duplicate XY samples.
        lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.r_[0., np.cumsum(lengths)]
        if cumulative[-1] > 1e-5:
            positions = np.linspace(.2, .8, min(5, max(1, int(cumulative[-1]/25))))*cumulative[-1]
            for distance in positions:
                j = min(int(np.searchsorted(cumulative, distance, side='right'))-1, len(lengths)-1)
                if lengths[j] <= 1e-5:
                    continue
                direction = (xy[j+1]-xy[j])/lengths[j]
                tip = xy[j]+direction*(distance-cumulative[j])
                tail = tip-direction*5
                normal = np.array([-direction[1], direction[0]])*2.3
                p, q = tail+normal, tail-normal
                parts.append(f'<path data-step="{offset+j+2}" d="M {p[0]:.3f} {p[1]:.3f} L {tip[0]:.3f} {tip[1]:.3f} L {q[0]:.3f} {q[1]:.3f}" fill="none" stroke="#111827" stroke-width="1.4"/>')
        for i, (x, y, width) in enumerate(path):
            tooltip = html.escape(f'笔画 {label} / 点 {i+1}/{len(path)}: x={x:.3f}, y={y:.3f}, width={width:.3f} px')
            parts.append(f'<circle data-step="{offset+i+1}" cx="{x:.3f}" cy="{y:.3f}" r="1.35" fill="{color}"><title>{tooltip}</title></circle>')
        x, y = xy[0]
        parts.append(f'<g data-step="{offset+1}"><circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="white" stroke="#16a34a" stroke-width="2"/><text x="{x+5:.3f}" y="{y-5:.3f}" font-size="8" fill="#166534">{html.escape(str(label))}</text><title>起点 / 笔画 {html.escape(str(label))}</title></g>')
        x, y = xy[-1]
        parts.append(f'<path data-step="{offset+len(path)}" d="M {x-3:.3f} {y-3:.3f} L {x+3:.3f} {y+3:.3f} M {x-3:.3f} {y+3:.3f} L {x+3:.3f} {y-3:.3f}" stroke="#dc2626" stroke-width="2"><title>终点 / 共{len(path)}点</title></path>')
        offset += len(path)
    if not total:
        parts.append('<text x="128" y="128" text-anchor="middle" fill="#64748b" font-size="12">未预测到笔画</text>')
    return ''.join(parts)+'</svg>'


def pack_paths(paths):
    paths = checked_paths(paths)
    points = np.concatenate(paths) if paths else np.empty((0, 3), np.float32)
    offsets = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64)
    return points, offsets


def player(panels):
    return '<div data-player><div class="controls"><button type="button">播放</button><input type="range" min="0" max="1000" value="1000" aria-label="书写进度"><output>100%</output><select aria-label="播放速度"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></div><div class="panels">'+''.join(
        '<div class="panel"><h3>'+html.escape(title)+'</h3>'+svg+'</div>' for title, svg in panels)+'</div></div>'


def legend():
    return '<p><span class="start">绿色圆圈＝起点</span>；<span class="end">红色叉号＝终点</span>；黑色箭头＝书写方向；起点旁数字＝笔顺。悬停采样点查看点序号及像素XYW。</p><p class="note">中心线展示走向；粗细与覆盖见原有叠加图。播放按点序推进，不表示真实书写速度；GT与预测按各自点数比例播放，不是DTW对齐。各笔独立，不绘制抬笔连接线。</p>'


def image_uri(image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('cannot encode gallery image')
    return 'data:image/png;base64,'+base64.b64encode(encoded).decode('ascii')


def path_mask(paths):
    mask = np.zeros((256,256),np.uint8)
    for path in checked_paths(paths):
        xy = np.rint(path[:,:2]).astype(int)
        for i,(x,y) in enumerate(xy):
            cv2.circle(mask,(int(x),int(y)),max(1,min(128,round(float(path[i,2])/2))),1,-1)
            if i:
                width=max(1,min(256,round(float(path[i-1:i+1,2].mean()))))
                cv2.line(mask,tuple(xy[i-1]),(int(x),int(y)),1,width)
    return mask.astype(bool)


def coverage_image(image, paths):
    result=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    result[path_mask(paths)]=(60,60,220)
    return result


def mask_image(image, masks):
    result=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    colors=[(235,99,37),(234,51,147),(6,119,217),(178,145,8)]
    for k,mask in enumerate(masks):
        selected=np.asarray(mask)>=.5
        result[selected]=(result[selected]*.25+np.array(colors[k%len(colors)])*.75).astype(np.uint8)
    return result


def image_panels(pictures):
    return '<div class="panels coverage">'+''.join('<figure><figcaption>'+html.escape(label)+'</figcaption><img src="'+image_uri(image)+'" alt="'+html.escape(label,quote=True)+'"></figure>' for label,image in pictures)+'</div>'


def write_compact_gallery(folder, title, image, gt_paths, pred_paths, *, indices=None,
                          complete=True, gt_masks=None, pred_masks=None, metadata=None,
                          extra_arrays=None, save_data=True):
    """Two files per selected character: one whole-character HTML and one NPZ."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    has_gt=gt_paths is not None
    gt_paths,pred_paths=checked_paths(gt_paths or []),checked_paths(pred_paths)
    gt,gt_offsets=pack_paths(gt_paths);pred,pred_offsets=pack_paths(pred_paths)
    if save_data:
        np.savez_compressed(folder/'character_trajectory.npz',points=pred,stroke_offsets=pred_offsets,
                            ground_truth_xyw_px=gt,ground_truth_stroke_offsets=gt_offsets,
                            stroke_indices=np.asarray(indices if indices is not None else range(len(pred_paths)),np.int64),
                            image=image,image_size=256,complete_character=complete,**(extra_arrays or {}))
    body='<h1>'+html.escape(title)+'</h1>'+legend()
    body+='<p>'+('完整输入字图' if complete else '部分笔画')+('；GT '+str(len(gt_paths))+'笔/'+str(len(gt))+'点' if has_gt else '；无GT标签')+f'，预测 {len(pred_paths)}笔/{len(pred)}点。</p>'
    pictures=[('原图',image)]
    if has_gt:
        pictures.append(('GT整字轨迹覆盖',coverage_image(image,gt_paths)))
    pictures.append(('预测整字轨迹覆盖',coverage_image(image,pred_paths)))
    body+=image_panels(pictures)
    if gt_masks is not None and pred_masks is not None:
        body+='<section><h2>笔画层 · 整字mask与笔顺</h2>'+image_panels([('GT笔画mask',mask_image(image,gt_masks)),('预测笔画mask',mask_image(image,pred_masks))])+'</section>'
    panels=[]
    if has_gt:
        panels.append(('GT整字走向',trajectory_svg(gt_paths,indices and [i+1 for i in indices])))
    panels.append(('预测整字走向',trajectory_svg(pred_paths,indices and [i+1 for i in indices])))
    body+='<section><h2>点层 · 整字走向与笔顺</h2>'+player(panels)+'</section>'
    body+='<p><a href="'+('character_trajectory.npz' if save_data else 'prediction.npz')+'">下载整字轨迹NPZ（像素XYW与笔画offsets）</a></p>'
    if metadata:
        body+='<details><summary>指标与评估条件</summary><pre>'+html.escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></details>'
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>'+STYLE+body+SCRIPT,encoding='utf-8')
    return dict(strokes=len(pred_paths),predicted_points=len(pred),complete_character=complete)


def write_character_gallery(folder, title, rows=None, expected_strokes=None):
    """Explicit legacy refresh only; never required by new evaluations."""
    folder=Path(folder);records=[]
    for path in sorted(folder.glob('stroke[0-9]*.npz')):
        with np.load(path,allow_pickle=False) as data:
            records.append((int(path.stem[6:]),data['ground_truth_xyw_px'].copy(),data['predicted_xyw_px'].copy()))
    if not records:
        raise ValueError(f'no saved stroke trajectories: {folder}')
    indices=[r[0] for r in records]
    image=cv2.imread(str(folder/'original.png'),cv2.IMREAD_GRAYSCALE)
    if image is None:
        image=np.full((256,256),255,np.uint8)
    return write_compact_gallery(folder,title,image,[r[1] for r in records],[r[2] for r in records],
                                 indices=indices,complete=expected_strokes is not None and indices==list(range(expected_strokes)),metadata=rows)


def write_evaluation_index(output, summary, titles=None):
    output=Path(output)
    body='<h1>整字覆盖与轨迹方向</h1>'+legend()
    body+='<details><summary>评估汇总</summary><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>'
    for sid in summary.get('visualization_ids',[]):
        folder=output/sid
        if not (folder/'index.html').exists():
            continue
        safe=html.escape(sid,quote=True)
        body+='<section><h2><a href="'+safe+'/index.html">'+html.escape((titles or {}).get(sid,sid))+' · 整字覆盖、方向与播放</a></h2>'
        with np.load(folder/'character_trajectory.npz',allow_pickle=False) as data:
            image=data['image'];points=data['points'];offsets=data['stroke_offsets']
            gt=data['ground_truth_xyw_px'];gt_offsets=data['ground_truth_stroke_offsets']
            pred_paths=[points[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
            gt_paths=[gt[a:b] for a,b in zip(gt_offsets[:-1],gt_offsets[1:])]
            body+=image_panels([('原图',image),('GT整字覆盖',coverage_image(image,gt_paths)),('预测整字覆盖',coverage_image(image,pred_paths))])
        body+='</section>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Trajectory evaluation</title>'+STYLE+body,encoding='utf-8')


def write_prediction_gallery(output, image, paths):
    return write_compact_gallery(output,'两阶段整字预测',image,None,paths,save_data=False,
                                 metadata=dict(condition='image-only; stroke order/masks/starts predicted by stage one'))
