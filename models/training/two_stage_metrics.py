"""Full-cascade metrics: penalize count errors and label aligned-only geometry."""
import numpy as np

from models.training.single_stroke_evaluation import raster, iou


ALIGNED_KEYS=('aligned_xy_mae_px','aligned_width_mae_px','aligned_width_bias_px',
              'aligned_endpoint_l2_px','aligned_point_count_mae','aligned_exact_point_count_rate')


def compare_trajectories(gt_paths, paths):
    pred_paths=[p['points'].detach().cpu().numpy()*np.array([255,255,256],np.float32) for p in paths]
    gt_masks=[raster(p) for p in gt_paths];pred_masks=[raster(p) for p in pred_paths]
    def union(masks):
        return np.logical_or.reduce(masks) if masks else np.zeros((256,256),bool)
    n,m=len(gt_paths),len(pred_paths);common=min(n,m)
    row=dict(character_trajectory_iou=iou(union(gt_masks),union(pred_masks)),
             ordered_trajectory_iou=sum(iou(gt_masks[k],pred_masks[k]) for k in range(common))/max(n,m,1),
             aligned_strokes=common,predicted_strokes=m,missing_strokes=max(n-m,0),extra_strokes=max(m-n,0),
             terminated=sum(bool(p['terminated']) for p in paths),truncated=sum(bool(p['truncated']) for p in paths))
    values={k:[] for k in ALIGNED_KEYS}
    for actual,pred in zip(gt_paths,pred_paths):
        t=np.linspace(0,1,max(len(actual),len(pred),2))
        def resample(points):
            return np.stack([np.interp(t,np.linspace(0,1,len(points)),points[:,j]) for j in range(3)],-1)
        error=resample(pred)-resample(actual)  # Include the predicted first point.
        values['aligned_xy_mae_px'].append(float(np.abs(error[:,:2]).mean()))
        values['aligned_width_mae_px'].append(float(np.abs(error[:,2]).mean()))
        values['aligned_width_bias_px'].append(float(error[:,2].mean()))
        values['aligned_endpoint_l2_px'].append(float(np.linalg.norm(pred[-1,:2]-actual[-1,:2])))
        values['aligned_point_count_mae'].append(abs(len(pred)-len(actual)))
        values['aligned_exact_point_count_rate'].append(float(len(pred)==len(actual)))
    row.update({k:float(np.mean(v)) if v else None for k,v in values.items()})
    return row,pred_paths


def summarize(rows):
    total=len(rows);aligned=sum(r['aligned_strokes'] for r in rows)
    report={k:sum(r[k] for r in rows)/total for k in ('character_trajectory_iou','ordered_trajectory_iou')}
    report.update({k:sum(r[k] for r in rows) for k in ('aligned_strokes','predicted_strokes','missing_strokes','extra_strokes','terminated','truncated')})
    report.update({k:sum((r[k] or 0)*r['aligned_strokes'] for r in rows)/aligned if aligned else None for k in ALIGNED_KEYS})
    report.update(terminated_rate=report['terminated']/report['predicted_strokes'] if report['predicted_strokes'] else None,
                  truncated_rate=report['truncated']/report['predicted_strokes'] if report['predicted_strokes'] else None,
                  condition='predicted masks and predicted starts; point model frozen',
                  alignment='stroke sequence index; aligned errors exclude missing/extra strokes; IoU penalizes count errors',
                  resampling='equal normalized sequence progress, includes predicted first point; not DTW')
    return report
