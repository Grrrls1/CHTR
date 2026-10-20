"""Normalize the character-organized chinese-calligraphy-dataset release."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import SCHEMA, digest, gray, identifier, output_dir, require, write_json, write_records


def normalize(item):
    source, character, output, margin = item
    image = gray(source)
    # This release contains white-background grayscale JPEGs. Refuse samples
    # whose corners do not support that convention instead of auto-inverting.
    corners = np.r_[image[:5, :5].ravel(), image[:5, -5:].ravel(),
                    image[-5:, :5].ravel(), image[-5:, -5:].ravel()]
    require(float(corners.mean()) > 127, 'unexpected dark background: ' + str(source))
    height, width = image.shape
    scale = (256 - 2 * margin) / float(max(height, width))
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (256 - new_width) // 2, (256 - new_height) // 2
    canvas = np.full((256, 256), 255, np.uint8)
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(
        image, (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    file_id = source.stem
    sample_id = identifier('ccd_u%05X_%s' % (ord(character), file_id))
    destination = output / (sample_id + '.png')
    require(cv2.imwrite(str(destination), canvas), 'cannot write ' + str(destination))
    return dict(
        schema=SCHEMA, sample_id=sample_id, character=character,
        image_path=str(destination.resolve()), source_image_path=str(source.resolve()),
        source_sha256=digest(source), source='chinese-calligraphy-dataset-221030',
        provenance_id='release-without-calligrapher-metadata',
        provenance_status='writer_and_style_unknown', pack_eligible=False,
        original_file_id=file_id, canvas=256,
        transform=dict(original_size=[width, height], resized_size=[new_width, new_height],
                       scale_xy=[new_width / width, new_height / height], padding=[left, top],
                       pixel_mapping='(x+0.5)*scale_xy-0.5+padding'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Directory containing label_character.csv and chinese-calligraphy-dataset/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    require(0 <= args.margin < 128 and args.workers > 0, 'invalid margin/workers')
    root = Path(args.dataset).resolve()
    image_root = root / 'chinese-calligraphy-dataset'
    label_path = root / 'label_character.csv'
    require(image_root.is_dir() and label_path.is_file(), 'unexpected dataset layout')
    with label_path.open(encoding='utf-8-sig', newline='') as handle:
        labels = list(csv.reader(handle))
    require(len(labels) == 7328 and all(len(row) == 2 for row in labels), 'unexpected label table')
    characters = [row[0] for row in labels]
    require(len(set(characters)) == len(characters) and all(len(c) == 1 for c in characters), 'invalid/duplicate characters')
    output = output_dir(args.output)
    jobs = []
    seen_names = set()
    for character in characters:
        directory = image_root / character
        require(directory.is_dir(), 'missing character directory: ' + character)
        for source in sorted(directory.glob('*.jpg')):
            require(source.stem not in seen_names, 'duplicate global image ID: ' + source.stem)
            seen_names.add(source.stem)
            jobs.append((source, character, output, args.margin))
    require(len(jobs) == 138499, 'unexpected image count: %d' % len(jobs))
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(job, pool.submit(normalize, job)) for job in jobs]
        for index, (job, future) in enumerate(futures, 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(dict(source_image_path=str(job[0]), character=job[1], error=str(exc)))
            if index % 5000 == 0:
                print('processed %d/%d' % (index, len(jobs)), flush=True)
    rows.sort(key=lambda row: int(row['original_file_id']))
    write_records(output / 'manifest.jsonl', rows)
    write_records(output / 'errors.jsonl', errors)
    write_json(output / 'summary.json', dict(
        dataset='chinese-calligraphy-dataset-221030', images=len(rows), errors=len(errors),
        characters=len({row['character'] for row in rows}), pack_eligible=False,
        limitation='This character-organized archive does not expose calligrapher/style identity.',
        archive_sha256='60f3a853373415a66a5902298c4ab1951ff0139580d114d0ab3e4d8248fcf235'))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
