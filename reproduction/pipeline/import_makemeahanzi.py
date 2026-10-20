"""Import official Make Me A Hanzi graphics.txt as ordered unlabeled stroke candidates."""
import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image

from common import SCHEMA, digest, identifier, output_dir, require, write_json, write_records


def render_record(record, output):
    character = record['character']
    strokes = record['strokes']
    medians = record['medians']
    require(len(character) == 1 and strokes and len(strokes) == len(medians), 'invalid graphic record')
    count = len(strokes)
    groups = ['<g transform="translate(%d,0) scale(1,-1) translate(0,-900)"><path d="%s" fill="black"/></g>' %
              (i * 1024, path.replace('&', '&amp;').replace('"', '&quot;'))
              for i, path in enumerate(strokes)]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="1024" viewBox="0 0 %d 1024">%s</svg>' %
           (count * 1024, count * 1024, ''.join(groups))).encode()
    rendered = cairosvg.svg2png(bytestring=svg, output_width=count * 256, output_height=256)
    rgba = np.asarray(Image.open(io.BytesIO(rendered)).convert('RGBA'))
    masks = np.stack([(rgba[:, i * 256:(i + 1) * 256, 3] > 0).astype(np.uint8) for i in range(count)])
    require(np.all(masks.reshape(count, -1).sum(1) > 0), 'empty rendered stroke')
    sample_id = identifier('mmah_u%05X' % ord(character))
    destination = output / (sample_id + '.npz')
    payload = dict(character=np.array(character), ordered_stroke_masks=masks,
                   stroke_order=np.arange(count, dtype=np.int64),
                   stroke_label=np.full(count, -1, dtype=np.int64),
                   category_system=np.array('unlabeled'),
                   category_status=np.array('no_stroke_categories_in_source'),
                   source=np.array('skishore/makemeahanzi graphics.txt'))
    for i, median in enumerate(medians):
        payload['median_%03d' % i] = np.asarray([[x / 4, (900 - y) / 4] for x, y in median], np.float32)
    np.savez_compressed(str(destination), **payload)
    preview = output / (sample_id + '.png')
    require(cv2.imwrite(str(preview), (1 - np.any(masks, axis=0).astype(np.uint8)) * 255), 'cannot write preview')
    return dict(schema=SCHEMA, sample_id=sample_id, character=character,
                ordered_strokes_path=str(destination.resolve()),
                ordered_strokes_sha256=digest(destination), preview_path=str(preview.resolve()),
                stroke_count=count, category_system='unlabeled',
                category_status='no_stroke_categories_in_source',
                order_source='Make Me A Hanzi', source='skishore/makemeahanzi graphics.txt',
                strokeextraction_24_compatible=False, review_status='pending')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphics', required=True, help='Official graphics.txt JSONL')
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--source-commit', required=True)
    args = p.parse_args()
    require(args.workers > 0 and len(args.source_commit) == 40, 'invalid workers/source commit')
    graphics_path = Path(args.graphics).resolve()
    records = [json.loads(s) for s in graphics_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    require(len({r['character'] for r in records}) == len(records), 'duplicate characters')
    output = output_dir(args.output)
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(record, pool.submit(render_record, record, output)) for record in records]
        for index, (record, future) in enumerate(futures, 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(dict(character=record.get('character'), error=str(exc)))
            if index % 500 == 0:
                print('processed %d/%d' % (index, len(records)), flush=True)
    rows.sort(key=lambda r: ord(r['character']))
    write_records(output / 'manifest.jsonl', rows)
    write_records(output / 'errors.jsonl', errors)
    write_json(output / 'summary.json', dict(source='https://github.com/skishore/makemeahanzi',
               source_commit=args.source_commit, graphics_sha256=digest(graphics_path),
               records=len(records), generated=len(rows), errors=len(errors),
               total_strokes=sum(r['stroke_count'] for r in rows),
               category_system='unlabeled',
               note='Native stroke order/medians; no CCSE or StrokeExtraction class labels.'))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
