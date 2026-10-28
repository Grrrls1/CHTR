"""Stage-one character samples, preserving the stage-two coordinate system."""
import numpy as np
import torch
from torch.utils.data import Dataset


class OrderedStrokeDataset(Dataset):
    def __init__(self, catalog, split, max_strokes=64):
        if split not in ('train', 'val', 'test') or max_strokes < 1:
            raise ValueError('invalid split or stroke cap')
        self.catalog, self.split, self.max_strokes = catalog, split, max_strokes
        self.ids = list(catalog.split[split + '_ids'])
        if not self.ids:
            raise ValueError('empty character split')
        for sid in self.ids:
            if not 1 <= catalog.rows[sid]['stroke_count'] <= max_strokes:
                raise ValueError(sid + ': stroke count exceeds cap; no truncation')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        image, points, offsets, masks = self.catalog.character_assets(sid)
        starts = points[offsets[:-1]].copy()
        starts[:, :2] /= 255.
        starts[:, 2] /= 256.
        boxes = []
        for mask in masks:
            y, x = np.where(mask)
            # Inclusive pixel-center XYXY; single-pixel masks may have zero area.
            boxes.append([x.min()/255., y.min()/255., x.max()/255., y.max()/255.])
        return dict(image=torch.from_numpy(image.astype(np.float32)[None]/255.),
                    masks=torch.from_numpy(masks.astype(np.float32)),
                    boxes=torch.tensor(boxes, dtype=torch.float32),
                    starts=torch.from_numpy(starts), sample_id=sid,
                    character=self.catalog.rows[sid]['character'])


def collate_ordered_strokes(samples):
    """Output slot j predicts stroke j; slot K predicts invalid, then padding."""
    counts = torch.tensor([len(s['masks']) for s in samples], dtype=torch.long)
    size = int(counts.max()) + 1
    b, h, w = len(samples), *samples[0]['image'].shape[-2:]
    masks = torch.zeros(b, size, h, w)
    history = torch.zeros(b, size-1, h, w)
    boxes = torch.zeros(b, size, 4)
    starts = torch.zeros(b, size, 3)
    for i, sample in enumerate(samples):
        n = int(counts[i])
        masks[i, :n] = sample['masks']
        history[i, :n] = sample['masks']
        boxes[i, :n] = sample['boxes']
        starts[i, :n] = sample['starts']
    slots = torch.arange(size)[None]
    return dict(images=torch.stack([s['image'] for s in samples]), history_masks=history,
                history_active=torch.arange(size-1)[None] < counts[:, None],
                masks=masks, boxes=boxes, starts=starts, counts=counts,
                valid=slots < counts[:, None], active=slots <= counts[:, None],
                keys=[s['sample_id'] for s in samples])
