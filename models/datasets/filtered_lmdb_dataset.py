"""Training-time whole-character IoU selection without changing stored records."""
import json
import math

import lmdb

from .lmdb_dataset import CHTRModelDataset


MASK_STAGES = ('mask_cleanup_whole', 'mask_extraction_whole')


def threshold(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(name + ' must be a number in [0,1] or null')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + ' must be a number in [0,1] or null')
    return value


def score_passes(value, minimum):
    if minimum is None:
        return True
    if value is None or isinstance(value, bool):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and 0 <= value <= 1 and value >= minimum


class CHTRFilteredDataset(CHTRModelDataset):
    def __init__(self, lmdb_dir, min_trajectory_iou=None, min_mask_iou=None,
                 mask_iou_stage='mask_cleanup_whole', **kwargs):
        min_trajectory_iou = threshold(min_trajectory_iou, 'min_trajectory_iou')
        min_mask_iou = threshold(min_mask_iou, 'min_mask_iou')
        if mask_iou_stage not in MASK_STAGES:
            raise ValueError('mask_iou_stage must be one of ' + str(MASK_STAGES))
        super().__init__(lmdb_dir, **kwargs)
        self.iou_filters = dict(min_trajectory_iou=min_trajectory_iou,
                                min_mask_iou=min_mask_iou, mask_iou_stage=mask_iou_stage)
        ready = len(self.ids)
        rejected_trajectory = rejected_mask = 0
        if min_trajectory_iou is not None or min_mask_iou is not None:
            selected = []
            # Only decode metadata, one record at a time; never load images/NPZs here.
            with lmdb.open(self.path, readonly=True, lock=False, readahead=False) as env:
                with env.begin() as txn:
                    for sid in self.ids:
                        metrics = json.loads(txn.get(sid.encode())).get('stage_iou') or {}
                        trajectory_ok = score_passes(metrics.get('trajectory_whole'), min_trajectory_iou)
                        mask_ok = score_passes(metrics.get(mask_iou_stage), min_mask_iou)
                        rejected_trajectory += not trajectory_ok
                        rejected_mask += not mask_ok
                        if trajectory_ok and mask_ok:
                            selected.append(sid)
            self.ids = selected
        self.selection_summary = dict(total=len(self.all_ids), pending=self.pending_count,
                                      trajectory_ready=ready, selected=len(self.ids),
                                      quality_excluded=ready-len(self.ids),
                                      rejected_trajectory_iou=rejected_trajectory,
                                      rejected_mask_iou=rejected_mask, filters=self.iou_filters,
                                      note='Per-metric rejection counts may overlap.')
        if not self.ids:
            raise ValueError('No samples remain after whole-character IoU selection: ' +
                             json.dumps(self.selection_summary))
