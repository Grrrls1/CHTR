"""Write image/mask/directed trajectory previews with labels to a local HTML page."""
import argparse
import html
from itertools import islice
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT.parent))
from models.datasets.lmdb_dataset import CHTRModelDataset, denormalize_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=str(MODEL_ROOT.parent/'data/chtr_mmah_lmdb_1915'))
    parser.add_argument('--output', default=str(MODEL_ROOT/'runs/data_preview'))
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of samples to render')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset = CHTRModelDataset(args.dataset, augment=args.augment)
    rows = []
    for sample in islice(dataset, args.limit):
        sid = sample['sample_id']
        image = np.rint(sample['image'][0].numpy()*255).astype(np.uint8)
        cv2.imwrite(str(out/(sid+'.png')), image)
        points = denormalize_points(sample['points'].numpy(), sample['image_size'])
        combined = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        panels = ['<img src="%s.png" alt="input">' % sid]
        for i, (a, b) in enumerate(zip(sample['stroke_offsets'][:-1], sample['stroke_offsets'][1:])):
            path = points[int(a):int(b)]
            mask = sample['stroke_masks'][i].numpy()
            panel = cv2.cvtColor(np.rint((1-mask)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            color = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[i*37%180, 220, 200]]]), cv2.COLOR_HSV2BGR)[0, 0])
            for t in range(len(path)-1):
                x, y = np.rint(path[t, :2]).astype(int)
                width = max(1, int(round(float(path[t, 2]))))
                cv2.circle(combined, (x, y), max(1, width//2), color, -1)
                if t % 10 == 0:
                    q = path[min(t+5, len(path)-1), :2]
                    cv2.arrowedLine(panel, (x, y), tuple(np.rint(q).astype(int)), color, 1, tipLength=0.4)
            for canvas in (combined, panel):
                cv2.circle(canvas, tuple(np.rint(path[0, :2]).astype(int)), 3, (0, 180, 0), -1)
                cv2.circle(canvas, tuple(np.rint(path[-1, :2]).astype(int)), 3, (0, 0, 255), -1)
            filename = '%s_stroke%03d.png' % (sid, i)
            cv2.imwrite(str(out/filename), panel)
            name = dataset.names[int(sample['stroke_types'][i])]
            panels.append('<figure><img src="%s"><figcaption>%d: %s; IoU %.4f</figcaption></figure>' %
                          (filename, i+1, html.escape(name), float(sample['stroke_quality'][i])))
        cv2.imwrite(str(out/(sid+'_trajectory.png')), combined)
        panels.insert(1, '<img src="%s_trajectory.png" alt="width trajectory">' % sid)
        rows.append('<section><h2>%s · %s</h2><div>%s</div></section>' % (html.escape(sample['character']), sid, ''.join(panels)))
    dataset.close()
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CHTR 数据检查</title>'
                                '<style>body{font-family:sans-serif}section div{display:flex;flex-wrap:wrap;gap:12px}'
                                'figure{margin:0}img{width:256px;height:256px;border:1px solid #ccc}</style>'
                                '<h1>逐笔掩码与轨迹检查</h1><p>绿色为起点，红色为终点，箭头表示方向。</p>'+''.join(rows), encoding='utf-8')
    print(out/'index.html')


if __name__ == '__main__':
    main()
