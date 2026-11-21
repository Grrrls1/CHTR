"""CPU aggregation and acceptance. Noise draws are collapsed within each instance first."""
import copy
import json
from pathlib import Path
import numpy as np
from models.experiments import mask_start_core as c
from models.scripts.evaluate_cross_domain_sampled import aggregate as aggregate_whole,mean_std


def records(ex,split,cond,draw):
    rows=[]
    for sid in ex.ids(split,cond['domain']):
        path=ex.e2_path(split,cond['domain'],cond['stroke'],cond['point'],sid) if cond['experiment']=='E2' else ex.e1_path(split,cond,draw,sid)
        r=ex.cached(path)
        if r is None:raise ValueError('missing/uncommitted expected cache: '+str(path))
        assert r['first_point_exact']
        rows.append(r)
    return rows


def collapse(draw_records):
    """One mean record per ID; first average draw values per matched instance."""
    by={}
    for rows in draw_records:
        for r in rows:by.setdefault(r['sample_id'],[]).append(r)
    out=[]
    for sid,rs in by.items():
        r=copy.deepcopy(rs[0]);r['draw']='mean_draws';r['noise_draws_averaged']=len(rs)
        for key in ['conditional_point_gpu_seconds','point_gpu_seconds','metric_cpu_seconds']:
            if key in r:r[key]=c.mean_defined([q.get(key) for q in rs])
        field='whole_character' if 'whole_character' in r else 'metrics'
        for key,value in r[field].items():
            if value is None or isinstance(value,(int,float)):r[field][key]=c.mean_defined([q[field][key] for q in rs])
            else:assert all(q[field][key]==value for q in rs)
        if 'instances' in r:
            for i,inst in enumerate(r['instances']):
                source=[q['instances'][i] for q in rs]
                assert all((q['pred_index'],q['gt_index'])==(inst['pred_index'],inst['gt_index']) for q in source)
                for scope in ['full','post_start']:
                    for key in inst[scope]:inst[scope][key]=c.mean_defined([q[scope][key] for q in source])
                for key in ['trajectory_iou','first_xy_l2_px','first_w_error_px','predicted_points','gt_points','clean_points','clean_point_truncated','clean_point_terminated','post_start_valid','predicted_first_only','gt_first_only','point_truncated','point_terminated']:
                    inst[key]=c.mean_defined([q[key] for q in source])
                for key,value in inst['arc'].items():
                    if key=='s':continue
                    if key.startswith('degenerate'):inst['arc'][key]=any(q['arc'][key] for q in source)
                    elif isinstance(value,list):inst['arc'][key]=np.mean([q['arc'][key] for q in source],axis=0).tolist()
                    else:inst['arc'][key]=float(np.mean([q['arc'][key] for q in source]))
            r['single_character']=c.single_character_metrics(r['instances'])
        out.append(r)
    return out


def summarize(rows):
    whole=[r.get('whole_character',r.get('metrics')) for r in rows]
    out=aggregate_whole(whole);out['image_char_iou']=float(np.mean([r['image_char_iou'] for r in whole]))
    out['characters']=len(rows)
    out['metric_cpu_seconds']=c.mean_defined([r.get('metric_cpu_seconds') for r in rows])
    out['conditional_point_gpu_seconds']=c.mean_defined([r.get('conditional_point_gpu_seconds') for r in rows])
    out['point_gpu_seconds']=c.mean_defined([r.get('point_gpu_seconds') for r in rows])
    if 'instances' not in rows[0]:return out
    eligible=[r for r in rows if r['instances']]
    out.update(matched_characters=len(eligible),matched_instances=sum(len(r['instances']) for r in rows),
        gt_strokes_total=sum(r['coverage']['gt_strokes'] for r in rows),pred_strokes_total=sum(r['coverage']['pred_strokes'] for r in rows))
    out['post_start_defined_characters']=sum(r['single_character']['post_start_any_valid_instances']>0 for r in rows)
    out['post_start_any_valid_instances']=sum(r['single_character']['post_start_any_valid_instances'] for r in rows)
    out['post_start_valid_instance_draw_fraction']=sum(r['single_character']['post_start_valid_draw_weight'] for r in rows)/max(out['matched_instances'],1)
    out['predicted_first_only_instance_draw_fraction']=sum(r['single_character']['predicted_first_only_draw_weight'] for r in rows)/max(out['matched_instances'],1)
    out['gt_match_coverage']=out['matched_instances']/max(out['gt_strokes_total'],1)
    out['pred_match_coverage']=out['matched_instances']/max(out['pred_strokes_total'],1)
    scalar=list(eligible[0]['single_character']) if eligible else []
    for key in scalar:
        if isinstance(eligible[0]['single_character'][key],list):continue
        out['single_'+key]=c.mean_defined([r['single_character'][key] for r in eligible])
    # Squared responses: draws averaged within instance, then instances within character, then characters.
    means={}
    for key in ['xy_clean_sq','w_clean_sq','xy_gt_sq','w_gt_sq','clean_xy_gt_sq','clean_w_gt_sq']:
        means[key]=np.mean([r['single_character'][key] for r in eligible],axis=0) if eligible else np.zeros(6)
    dx=c.mean_defined([r['single_character']['actual_delta_xy_sq'] for r in eligible]) or 0
    dw=c.mean_defined([r['single_character']['actual_delta_w_sq'] for r in eligible]) or 0
    for j,s in enumerate(c.POSITIONS):
        prefix=f'arc_s{s:g}_'
        for label,key in [('Dxy','xy_clean_sq'),('Dw','w_clean_sq'),('GTxy','xy_gt_sq'),('GTw','w_gt_sq'),('cleanGTxy','clean_xy_gt_sq'),('cleanGTw','clean_w_gt_sq')]:
            out[prefix+label]=float(np.sqrt(max(means[key][j],0))) if eligible else None
        out[prefix+'Rxy']=float(np.sqrt(means['xy_clean_sq'][j]/dx)) if dx>0 else None
        out[prefix+'Rw']=float(np.sqrt(means['w_clean_sq'][j]/dw)) if dw>0 else None
    return out


def compare_effects(data,ids,effects,fields,meta):
    result=[]
    for label,coeff in effects.items():
        for field in fields:
            vals=[]
            for sid in ids:
                values=[data[key][sid].get(field) for key in coeff]
                if all(v is not None for v in values):vals.append(sum(data[key][sid][field]*w for key,w in coeff.items()))
            result.append(dict(**meta,effect=label,metric=field,paired_characters=len(vals),difference=c.mean_defined(vals)))
    return result


def noise_diagnostics(rs,meta):
    inst=[i for r in rs for i in r.get('instances',[]) if 'noise' in i]
    if not inst:return dict(**meta,matched_instances=0)
    request=np.asarray([i['noise']['requested_noise_px'] for i in inst]);actual=np.asarray([i['noise']['actual_noise_px'] for i in inst])
    out=dict(**meta,matched_instances=len(inst),xy_clipped_rate=np.mean([i['noise']['xy_clipped'] for i in inst]),
             w_floored_rate=np.mean([i['noise']['w_floored'] for i in inst]))
    for label,a in [('requested',request),('actual',actual)]:
        for j,comp in enumerate(['x','y','w']):
            out[label+'_'+comp+'_mean_px']=float(a[:,j].mean());out[label+'_'+comp+'_variance_px2']=float(a[:,j].var())
    return out


def noise_acceptance(ex,split):
    for partition in range(12):assert c.read_json(ex.out/'acceptance'/f'{split}_noise_partition_{partition}.json')['state']=='passed'
    checked=0
    for cond in ex.cfg['conditions']:
        if cond['draws']!=5:continue
        for draw in range(5):
            rs=records(ex,split,cond,draw)
            for r in rs:
                assert r['noise_calibration_sha256']==c.digest(c.read_json(ex.out/'noise_calibration.json'))
                assert r['coverage']['matching_identity']==c.digest(ex.match(split,r['sample_id']))
                # Across mask levels, reuse the exact requested and clipped start, never a projected start.
                other=dict(cond,mask='PRED' if cond['mask']=='GT' else 'GT')
                other['condition_id']=f"E1_{cond['point']}_{other['mask']}_{cond['mode']}_k{cond['k']:g}"
                alt=ex.cached(ex.e1_path(split,other,draw,r['sample_id']));assert alt is not None
                for a,b in zip(r['instances'],alt['instances']):
                    assert a['input_first_xyw_px']==b['input_first_xyw_px'] and a['noise']==b['noise']
                    delta=np.asarray(a['noise']['actual_noise_px'])
                    assert np.isclose(a['arc']['xy_clean_sq'][0],float(np.sum(delta[:2]**2)),rtol=1e-4,atol=1e-3)
                    assert np.isclose(a['arc']['w_clean_sq'][0],float(delta[2]**2),rtol=1e-4,atol=1e-3)
                checked+=1
    c.write_json(ex.out/'acceptance'/f'{split}_noise.json',dict(state='passed',records=checked,units=180,
                 matching_fixed=True,cross_mask_noise_identical=True,first_point_preserved=True))


def plots_and_examples(ex,split,summaries,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions=ex.cfg['conditions']
    for point in ['BOOK','TRANSFER']:
        for mask in ['GT','PRED']:
            fig,axes=plt.subplots(2,3,figsize=(12,7))
            for col,mode in enumerate(['N10','N01','N11']):
                for k in c.LEVELS:
                    key=f'E1_{point}_{mask}_{mode}_k{k:g}'
                    if key not in summaries:continue
                    r=summaries[key]
                    for row,comp in enumerate(['xy','w']):
                        normalized=(comp=='xy' and mode!='N01') or (comp=='w' and mode!='N10')
                        field=('Rxy' if comp=='xy' else 'Rw') if normalized else ('Dxy' if comp=='xy' else 'Dw')
                        y=[r.get(f'arc_s{s:g}_'+field) for s in c.POSITIONS]
                        axes[row,col].plot(c.POSITIONS,y,marker='o',label=f'k={k:g}')
                        axes[row,col].set_title(mode+' '+field+(' ratio' if normalized else ' px'))
                        axes[row,col].set_xlabel('normalized geometric arc progress');axes[row,col].grid(alpha=.3)
                for row in range(2):axes[row,col].legend()
            fig.suptitle(f'{split} P_{point}, M_{mask}; draws averaged within instance; zero-length cases flagged')
            fig.tight_layout();dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/f'{split}_{point}_{mask}_response.png',dpi=150);plt.close(fig)
    # Four-condition mask/start overlay on all 8 predetermined smoke IDs; no selection by performance.
    for point in ['BOOK','TRANSFER']:
        for sid in ex.ids('smoke','BOOK'):
            image,gt,gmask=ex.assets('BOOK',sid);fig,axes=plt.subplots(2,2,figsize=(9,9))
            for ax,(mask,mode) in zip(axes.flat,[('GT','N00'),('PRED','N00'),('GT','PRED'),('PRED','PRED')]):
                cond=next(q for q in conditions if q['experiment']=='E1' and q['draws']==1 and q['point']==point and q['mask']==mask and q['mode']==mode)
                r=ex.cached(ex.e1_path('smoke',cond,0,sid))
                if not r:continue
                pred=c.paths_from(r);rgb=np.repeat(image[:,:,None],3,2).copy()
                gm=c.masks_union([c.raster(p) for p in gt]);pm=c.masks_union([c.raster(p) for p in pred])
                rgb[gm]=(.5*rgb[gm]+.5*np.array([0,180,0])).astype(np.uint8)
                rgb[pm]=(.5*rgb[pm]+.5*np.array([230,0,0])).astype(np.uint8)
                ax.imshow(rgb);ax.set_title('M_'+mask+' / '+mode);ax.axis('off')
            fig.suptitle(point+' '+sid+'; green GT / red prediction')
            fig.tight_layout();q=ex.out/'examples'/'smoke';q.mkdir(parents=True,exist_ok=True)
            fig.savefig(q/(point+'_'+c.digest(sid)[:16]+'.png'),dpi=130);plt.close(fig)


def aggregate(out,phase,report_path=None):
    from models.scripts.evaluate_mask_start_stage import Experiment
    ex=Experiment(out,gpu=False)
    try:
        if phase in ['val_noise','final']:noise_acceptance(ex,'val')
        if phase=='final':noise_acceptance(ex,'test')
        selected=[q for q in ex.cfg['conditions'] if phase!='base' or q['draws']==1]
        splits=['val'] if phase=='val_noise' else ['val','test']
        dest=out/'summaries'/phase;dest.mkdir(parents=True,exist_ok=True)
        units=[];repeats=[];means=[];noise_rows=[];execution=[];effect_rows=[];curve_rows=[]
        for split in splits:
            summary_unique={}
            e2data={};e1data={}
            for cond in selected:
                draw_records=[]
                for draw in range(cond['draws']):
                    rs=records(ex,split,cond,draw);draw_records.append(rs)
                    meta=dict(split=split,condition_id=cond['condition_id'],experiment=cond['experiment'],domain=cond['domain'],point=cond['point'],mask=cond['mask'],mode=cond['mode'],k=cond['k'],draw=draw)
                    summary=summarize(rs);units.append(dict(**meta,**summary))
                    noise_rows.append(noise_diagnostics(rs,meta))
                    execution.append(dict(**meta,state='complete',completed=len(rs),required=len(ex.ids(split,cond['domain'])),
                        oracle=cond['oracle'],cache_source='E2:S_TRANSFER' if cond['experiment']=='E1' and cond['mask']=='PRED' and cond['mode']=='PRED' else 'new conditional inference',
                        actual_point_stroke_inferences=0 if cond['experiment']=='E1' and cond['mask']=='PRED' and cond['mode']=='PRED' else sum(len(r.get('instances',c.paths_from(r))) for r in rs)))
                collapsed=collapse(draw_records);lookup={r['sample_id']:r for r in collapsed}
                summary_unique[cond['condition_id']]=summarize(collapsed)
                curve_rows.append(dict(split=split,condition_id=cond['condition_id'],noise_draws=cond['draws'],unique_characters=len(collapsed),**summary_unique[cond['condition_id']]))
                groups=[('val100',ex.ids(split,cond['domain']))] if split=='val' else [(str(seed),c.read_json(out/'manifests'/cond['domain']/f'seed_{seed}.json')['ids']) for seed in ex.cfg['repeats']]
                summaries=[]
                for repeat,ids in groups:
                    values=summarize([lookup[s] for s in ids]);summaries.append(values)
                    repeats.append(dict(split=split,condition_id=cond['condition_id'],repeat=repeat,noise_draws=cond['draws'],domain=cond['domain'],oracle=cond['oracle'],full_split=False,**values))
                means.append(dict(split=split,condition_id=cond['condition_id'],domain=cond['domain'],oracle=cond['oracle'],noise_draws=cond['draws'],**mean_std(summaries)))
                if cond['experiment']=='E2':
                    e2data.setdefault(cond['domain'],{})[cond['stroke'][0]+cond['point'][0]]={s:r['metrics'] for s,r in lookup.items()}
                    # JSONL per-character metrics, independent from heavy prediction files.
                    c.atomic_bytes(dest/f"E2_{split}_{cond['domain']}_{cond['stroke']}_{cond['point']}.jsonl",''.join(json.dumps(dict(sample_id=s,**v),allow_nan=False)+'\n' for s,v in e2data[cond['domain']][cond['stroke'][0]+cond['point'][0]].items()).encode())
                elif cond['experiment']=='E1' and cond['draws']==1:
                    data={}
                    for s,r in lookup.items():
                        v=dict(r['whole_character']);v.update({'single_'+k:x for k,x in r['single_character'].items() if not isinstance(x,list)})
                        data[s]=v
                    e1data.setdefault(cond['point'],{})[('0' if cond['mask']=='GT' else '1')+('0' if cond['mode']=='N00' else '1')]=data
            for domain,data in e2data.items():
                effects={'BT-BB':{'BT':1,'BB':-1},'TT-TB':{'TT':1,'TB':-1},'TB-BB':{'TB':1,'BB':-1},'TT-BT':{'TT':1,'BT':-1},'interaction':{'TT':1,'TB':-1,'BT':-1,'BB':1}}
                groups=[('val100',ex.ids(split,domain))] if split=='val' else [(str(seed),c.read_json(out/'manifests'/domain/f'seed_{seed}.json')['ids']) for seed in ex.cfg['repeats']]
                for repeat,ids in groups:effect_rows+=compare_effects(data,ids,effects,['aiou','image_char_iou','ordered_trajectory_iou','ldtw_px','stroke_count_mae'],dict(experiment='E2',split=split,domain=domain,repeat=repeat))
            for point,data in e1data.items():
                effects={'mask_at_GT_start':{'10':1,'00':-1},'mask_at_PRED_start':{'11':1,'01':-1},'start_at_GT_mask':{'01':1,'00':-1},'start_at_PRED_mask':{'11':1,'10':-1},'interaction':{'11':1,'10':-1,'01':-1,'00':1}}
                groups=[('val100',ex.ids(split,'BOOK'))] if split=='val' else [(str(seed),c.read_json(out/'manifests'/'BOOK'/f'seed_{seed}.json')['ids']) for seed in ex.cfg['repeats']]
                for repeat,ids in groups:effect_rows+=compare_effects(data,ids,effects,['image_char_iou','ordered_trajectory_iou','single_trajectory_iou','single_post_start_aligned_xy_mae_px','single_post_start_aligned_width_mae_px','single_post_start_aligned_endpoint_l2_px'],dict(experiment='E1',split=split,domain='BOOK',point=point,repeat=repeat))
            mask_comparison(ex,split,dest,phase!='base')
            if phase in ['val_noise','final']:plots_and_examples(ex,split,summary_unique,out/'figures')
        if phase=='base':baseline_reproduction(ex,dest)
        c.write_csv(dest/'per_execution_unit.csv',units);c.write_csv(dest/'per_repeat.csv',repeats)
        c.write_csv(dest/'mean_std.csv',means);c.write_csv(dest/'paired_effects.csv',effect_rows)
        c.write_csv(dest/'noise_requested_actual.csv',noise_rows);c.write_csv(dest/'response_unique_ids.csv',curve_rows)
        import csv
        manifest=list(csv.DictReader((out/'execution_manifest.csv').open()))
        updates={(r['split'],r['condition_id'],str(r['draw'])):r for r in execution}
        for r in manifest:
            key=(r['split'],r['condition_id'],str(r['draw']))
            if key in updates:r.update(updates[key])
        assert len(manifest)==396
        c.write_csv(out/'execution_manifest.csv',manifest)
        expected=36 if phase=='base' else (198 if phase=='val_noise' else 396)
        assert len(execution)==expected,(len(execution),expected)
        c.write_json(out/'acceptance'/f'{phase}_summary.json',dict(state='passed',execution_rows=len(execution),
            conditions_per_split=len(selected),response_protocol='mask_start_noise_response_arc64_v2',noise_draws_collapsed_within_instance=True))
        title=f'HiWTR E1 mask/首点与E2阶段交叉评估（{phase}）'
        lines=[title,'','状态：本文件只包含已逐字验收的条件；条件定义见condition_manifest.csv。',
            f'本阶段条件/域合并数：{len(selected)}；执行manifest行数：{len(execution)}。',
            '所有权重冻结；不训练；E1首点始终为完整XYW，模型原样保留。',
            'E1匹配单笔、全部GT单笔参照和混合oracle整字分别列出；不把GT条件当作部署成绩。',
            'E1的ordered_mask_iou等笔画层字段始终描述固定原始上游，不代表替换后的条件mask质量。',
            'E2全部为预测mask/首点的纯图像串联。计时为共享第一阶段与点层同步组件之和，不是重复连续测速。',
            '噪声尺度仅由固定BOOK val100校准；N10/N01/N11共享分量与强度z；跨mask不重新抽样。',
            '恢复曲线包含显式零弧长退化处理；必须同时查看GT误差、长度、点数与停止，不能仅用R<1认定纠错。','',
            'E2主要指标（test为5份100字清单均值±样本SD；val仅固定100字）','split / condition / ImageCharIoU / OrdIoU / LDTW / countMAE']
        for r in means:
            if r['condition_id'].startswith('E2_'):
                fmt=lambda k:f"{r[k+'_mean']:.6f} ± {r[k+'_sd']:.6f}" if r[k+'_sd'] is not None else str(r[k+'_mean'])
                lines.append(r['split']+' / '+r['condition_id']+' / '+' / '.join(fmt(k) for k in ['image_char_iou','ordered_trajectory_iou','ldtw_px','stroke_count_mae']))
        lines+=['','E1现实条件主要指标：split / condition / matched_instances(mean) / singleIoU / postXY / wholeImageCharIoU']
        for r in means:
            if r['condition_id'].startswith('E1_') and '_k' not in r['condition_id']:
                lines.append(' / '.join(str(r.get(k)) for k in ['split','condition_id','matched_instances_mean','single_trajectory_iou_mean','single_post_start_aligned_xy_mae_px_mean','image_char_iou_mean']))
        lines+=['','E1后续误差的配对影响（在每个repeat内同字配对；正负方向见指标定义）：']
        for point in ['BOOK','TRANSFER']:
            for effect in ['mask_at_GT_start','mask_at_PRED_start','start_at_GT_mask','start_at_PRED_mask','interaction']:
                for metric in ['single_trajectory_iou','single_post_start_aligned_xy_mae_px','single_post_start_aligned_width_mae_px']:
                    vals=[r['difference'] for r in effect_rows if r['experiment']=='E1' and r['split']==('val' if phase=='val_noise' else 'test') and r.get('point')==point and r['effect']==effect and r['metric']==metric and r['difference'] is not None]
                    if vals:lines.append(f'{point} / {effect} / {metric}: {np.mean(vals):+.6f}; defined repeats={len(vals)}')
        lines+=['','噪声恢复摘要：R小于1只表示相对同mask N00的影响衰减；须同时阅读GT误差和退化/长度/停止。']
        for r in curve_rows:
            if r['split']!=('val' if phase=='val_noise' else 'test') or '_k' not in r['condition_id']:continue
            lines.append(str({k:r.get(k) for k in ['condition_id','matched_instances','arc_s1_Rxy','arc_s1_Rw','arc_s1_GTxy','arc_s1_cleanGTxy','single_pred_length_px','single_clean_length_px','single_predicted_points','single_clean_points','single_point_truncated','single_clean_point_truncated','post_start_defined_characters','post_start_valid_instance_draw_fraction','predicted_first_only_instance_draw_fraction','single_degenerate_pred_count']}))
        lines+=['','详细表：mean_std.csv、per_repeat.csv、per_execution_unit.csv、paired_effects.csv。',
            '请求/实际噪声及裁剪：noise_requested_actual.csv；恢复曲线数值：response_unique_ids.csv；图：figures/。',
            '原生上限均64笔/400点，FP32、point batch16、阈值.5；不截断GT。',
            '各repeat可能重叠，draw是输入扰动重复而非训练种子；不将重复ID/draw当作独立字符。',
            '阶段交叉用于定位现有系统收益，历史预算/选优不同，不是严格预训练因果收益。']
        (dest/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        report=Path(report_path) if report_path is not None else Path(__file__).resolve().parents[1]/'reports'/f'HiWTR_mask_start_stage_{out.name}_{phase}.txt'
        report.write_text('\n'.join(lines)+f'\n\n原始输出：{out}\n')
        c.write_json(out/'status.json',dict(state='complete' if phase=='final' else 'partial_complete',phase=phase,
                  completed_execution_rows=sum(r['state']=='complete' for r in manifest),conditions_per_split=len(selected),report=str(report)))
        print('\n'.join(lines),flush=True)
    finally:ex.close()


def smoke_report(ex):
    summaries={};cost=[]
    for cond in ex.cfg['conditions']:
        rs=records(ex,'smoke',cond,0);summaries[cond['condition_id']]=summarize(rs)
        if cond['draws']==5:
            secs=sum((r['conditional_point_gpu_seconds'] or 0) for r in rs)/len(rs)
            cost.append(dict(condition_id=cond['condition_id'],mean_point_gpu_seconds_per_character=secs,
                estimate_val_5draw_seconds=secs*100*5,estimate_test_5draw_seconds=secs*483*5))
            other=dict(cond,mask='PRED' if cond['mask']=='GT' else 'GT')
            other['condition_id']=f"E1_{cond['point']}_{other['mask']}_{cond['mode']}_k{cond['k']:g}"
            for r in rs:
                alt=ex.cached(ex.e1_path('smoke',other,0,r['sample_id']));assert alt is not None
                for a,b in zip(r['instances'],alt['instances']):
                    assert a['input_first_xyw_px']==b['input_first_xyw_px'] and a['noise']==b['noise']
                    delta=np.asarray(a['noise']['actual_noise_px'])
                    assert np.isclose(a['arc']['xy_clean_sq'][0],np.sum(delta[:2]**2),rtol=1e-4,atol=1e-3)
                    assert np.isclose(a['arc']['w_clean_sq'][0],delta[2]**2,rtol=1e-4,atol=1e-3)
    c.write_csv(ex.out/'smoke_condition_metrics.csv',[dict(condition_id=k,**v) for k,v in summaries.items()])
    c.write_json(ex.out/'resource_estimate.json',dict(source='8 prespecified BOOK val images, one draw; provisional smoke calibration',
        uncertainty='noise changes stopping; estimates exclude rendering/metrics/cache and are not wall-time guarantees',
        noise_val_point_gpu_seconds=sum(r['estimate_val_5draw_seconds'] for r in cost),
        noise_test_point_gpu_seconds=sum(r['estimate_test_5draw_seconds'] for r in cost),conditions=cost))
    plots_and_examples(ex,'smoke',summaries,ex.out/'figures')
    smoke_noise_matrices(ex)


def mask_comparison(ex,split,dest,include_noise):
    """Actual trajectory differences across masks at identical starts, not differences of response magnitudes."""
    gt_cache={sid:ex.assets('BOOK',sid)[1] for sid in ex.ids(split,'BOOK')}
    selected=[q for q in ex.cfg['conditions'] if q['experiment']=='E1' and q['mask']=='GT' and (include_noise or q['draws']==1)]
    rows=[]
    for cond in selected:
        other=dict(cond,mask='PRED',condition_id=cond['condition_id'].replace('_GT_','_PRED_'))
        char_arrays=[];ninstances=0;degenerate=0
        for sid,gt in gt_cache.items():
            draws=[];matched=0
            for draw in range(cond['draws']):
                a=ex.cached(ex.e1_path(split,cond,draw,sid));b=ex.cached(ex.e1_path(split,other,draw,sid))
                assert a and b
                ap,bp=c.paths_from(a),c.paths_from(b);values=[]
                assert len(a['instances'])==len(b['instances'])
                for ia,ib in zip(a['instances'],b['instances']):
                    assert ia['input_first_xyw_px']==ib['input_first_xyw_px']
                    pid,gi=ia['pred_index'],ia['gt_index'];assert (pid,gi)==(ib['pred_index'],ib['gt_index'])
                    z=c.response(gt[gi],ap[pid],bp[pid],[0,0,0])
                    assert z['xy_clean_sq'][0]<1e-3 and z['w_clean_sq'][0]<1e-3
                    values.append(np.array([z[k] for k in ['xy_clean_sq','w_clean_sq','clean_xy_gt_sq','clean_w_gt_sq','xy_gt_sq','w_gt_sq']]))
                    degenerate+=int(z['degenerate_gt'] or z['degenerate_clean'] or z['degenerate_pred'])
                if values:draws.append(np.mean(values,axis=0));matched=len(values)
            if draws:char_arrays.append(np.mean(draws,axis=0));ninstances+=matched
        a=np.sqrt(np.mean(char_arrays,axis=0)) if char_arrays else np.full((6,6),np.nan)
        for j,s in enumerate(c.POSITIONS):
            vals=[float(a[i,j]) if np.isfinite(a[i,j]) else None for i in range(6)]
            rows.append(dict(split=split,point=cond['point'],mode=cond['mode'],k=cond['k'],noise_draws=cond['draws'],s=float(s),
                matched_characters=len(char_arrays),matched_instances=ninstances,degenerate_instance_draws=degenerate,
                mask_xy_difference_px=vals[0],mask_w_difference_px=vals[1],GTmask_xy_GT_rmse_px=vals[2],GTmask_w_GT_rmse_px=vals[3],
                PREDmask_xy_GT_rmse_px=vals[4],PREDmask_w_GT_rmse_px=vals[5]))
    c.write_csv(dest/f'{split}_same_start_mask_curves.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for point in ['BOOK','TRANSFER']:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for mode,k in sorted({(r['mode'],r['k']) for r in rows if r['point']==point}):
            selected=sorted([r for r in rows if r['point']==point and r['mode']==mode and r['k']==k],key=lambda r:r['s'])
            for ax,metric in zip(axes,['mask_xy_difference_px','mask_w_difference_px']):
                ax.plot([r['s'] for r in selected],[r[metric] for r in selected],label=f'{mode} k={k:g}')
                ax.set_xlabel('geometric arc progress');ax.set_ylabel(metric);ax.grid(alpha=.3)
        axes[-1].legend(fontsize=7);fig.suptitle(f'{split} P_{point}: identical start, PRED mask versus GT mask')
        fig.tight_layout();folder=ex.out/'figures';folder.mkdir(exist_ok=True)
        fig.savefig(folder/f'{split}_{point}_mask_difference.png',dpi=150);plt.close(fig)
    return rows


def smoke_noise_matrices(ex):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for point in ['BOOK','TRANSFER']:
        for sid in ex.ids('smoke','BOOK'):
            image,gt,gmask=ex.assets('BOOK',sid);stage=ex.load_stage('smoke','BOOK','TRANSFER',sid)
            gt_union=c.masks_union([c.raster(p) for p in gt]);gt_starts=np.array([p[0] for p in gt])
            for k in c.LEVELS:
                fig,axes=plt.subplots(2,5,figsize=(17,7))
                for row,mask in enumerate(['GT','PRED']):
                    for col,mode in enumerate(['N00','N10','N01','N11','PRED']):
                        cond=next(x for x in ex.cfg['conditions'] if x['experiment']=='E1' and x['point']==point and x['mask']==mask and x['mode']==mode and (x['draws']==1 or x['k']==k))
                        record=ex.cached(ex.e1_path('smoke',cond,0,sid));assert record
                        pred=c.paths_from(record);pm=c.masks_union([c.raster(p) for p in pred]);rgb=np.repeat(image[:,:,None],3,2).copy()
                        rgb[gt_union]=(.5*rgb[gt_union]+.5*np.array([0,180,0])).astype(np.uint8)
                        rgb[pm]=(.5*rgb[pm]+.5*np.array([230,0,0])).astype(np.uint8)
                        effective=stage['masks'].copy();starts=stage['starts']*c.SCALE
                        for inst in record['instances']:
                            pid,gi=inst['pred_index'],inst['gt_index'];starts[pid]=inst['input_first_xyw_px']
                            if mask=='GT':effective[pid]=gmask[gi]
                        ax=axes[row,col];ax.imshow(rgb)
                        if len(effective) and c.masks_union(effective).any():ax.contour(c.masks_union(effective),levels=[.5],colors=['orange'],linewidths=.4)
                        if len(starts):ax.scatter(starts[:,0],starts[:,1],s=12,c='magenta',marker='x')
                        ax.scatter(gt_starts[:,0],gt_starts[:,1],s=6,c='cyan',marker='.')
                        ax.set_title(f'M_{mask} / {mode}');ax.axis('off')
                fig.suptitle(f'{point} k={k:g} draw=0 {sid}; green GT, red output, orange input-mask boundary, magenta given starts')
                fig.tight_layout();folder=ex.out/'examples'/'smoke_noise';folder.mkdir(parents=True,exist_ok=True)
                fig.savefig(folder/f'{point}_{c.digest(sid)[:16]}_k{k:g}.png',dpi=100);plt.close(fig)


def baseline_reproduction(ex,dest):
    from models.scripts.evaluate_mask_start_stage import BASE
    rows=[]
    for domain in ['BOOK','CASIA']:
        for model in ['BOOK','TRANSFER']:
            for sid in ex.ids('test',domain):
                old=c.read_json(BASE/'predictions'/('M_'+model)/domain/(c.digest(sid)+'.json'))
                new=ex.cached(ex.e2_path('test',domain,model,model,sid));assert new
                a,b=c.paths_from(old),c.paths_from(new)
                same_shape=len(a)==len(b) and all(x.shape==y.shape for x,y in zip(a,b))
                max_xyw=max((float(np.max(np.abs(x-y))) for x,y in zip(a,b)),default=0.) if same_shape else None
                state_equal=old['prediction']['point_states']==[{k:v for k,v in st.items() if k!='first_point_exact'} for st in new['prediction']['point_states']]
                row=dict(domain=domain,model=model,sample_id=sid,same_shapes=same_shape,max_xyw_difference_px=max_xyw,
                         identical_point_states=state_equal,within_1e_4_px=bool(same_shape and max_xyw<=1e-4))
                for key in ['aiou','character_trajectory_iou','ordered_trajectory_iou','ldtw_px','stroke_count_mae']:
                    x,y=old['metrics'][key],new['metrics'][key]
                    row['delta_'+key]=y-x if x is not None and y is not None else None
                rows.append(row)
    c.write_csv(dest/'historical_BB_TT_reproduction.csv',rows)
    c.write_json(dest/'historical_BB_TT_reproduction.json',dict(cases=len(rows),same_shapes=sum(r['same_shapes'] for r in rows),
        within_1e_4_px=sum(r['within_1e_4_px'] for r in rows),identical_point_states=sum(r['identical_point_states'] for r in rows),
        interpretation='fresh 4-way execution, no historical prediction reuse; deviations are reported without test-based weight selection'))
