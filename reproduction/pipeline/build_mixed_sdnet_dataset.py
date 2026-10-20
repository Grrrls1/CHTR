"""Build a 30/20/50 RHSEDB/MMAH-CCSE/MMAH-augmented-CCSE SDNet dataset."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from common import digest, output_dir, records, require, write_json, write_records


LABEL_NAMES = [
    '点', '横', '竖', '撇', '提', '横折', '横撇/横钩', '横折钩',
    '竖弯钩', '横折弯钩/横斜钩', '竖折折钩', '横折折/横折弯',
    '横折提', '横折折折钩/横撇弯钩', '竖折撇/竖折折', '横折折撇',
    '捺', '斜钩', '弯钩', '竖钩', '竖折/竖弯', '撇折', '撇点', '竖提',
]


def seed_for(*values):
    return int(hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:16], 16)


def decode_annotation(annotation, height, width):
    segmentation = annotation['segmentation']
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        result = mask_utils.decode(mask_utils.merge(rles))
    elif isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), list):
        result = mask_utils.decode(mask_utils.frPyObjects(segmentation, height, width))
    else:
        result = mask_utils.decode(segmentation)
    if result.ndim == 3:
        result = np.any(result, axis=2)
    return result.astype(np.uint8)


def annotation_has_mask(annotation, height, width):
    segmentation = annotation.get('segmentation')
    if not isinstance(segmentation, (list, dict)):
        return False
    # Some source polygons are syntactically nonempty but rasterize to zero area,
    # and some crowd RLEs encode only background. Validate the actual raster.
    return bool(np.any(decode_annotation(annotation, height, width)))


def resize_mask(mask):
    return (cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def augment_masks(masks, character, variant):
    rng = np.random.default_rng(seed_for('mmah-ccse-aug-v1', character, variant))
    angle = float(rng.uniform(-10, 10))
    scale_x, scale_y = rng.uniform(.88, 1.12, size=2)
    translate_x, translate_y = rng.uniform(-12, 12, size=2)
    theta = np.deg2rad(angle)
    cosine, sine = np.cos(theta), np.sin(theta)
    matrix = np.array([[scale_x * cosine, -scale_y * sine, translate_x],
                       [scale_x * sine, scale_y * cosine, translate_y]], np.float32)
    center = np.array([127.5, 127.5], np.float32)
    matrix[:, 2] += center - matrix[:, :2] @ center

    grid_x, grid_y = np.meshgrid(np.arange(256, dtype=np.float32),
                                 np.arange(256, dtype=np.float32))
    coarse_x = rng.normal(0, 3.0, (9, 9)).astype(np.float32)
    coarse_y = rng.normal(0, 3.0, (9, 9)).astype(np.float32)
    displacement_x = cv2.resize(coarse_x, (256, 256), interpolation=cv2.INTER_CUBIC)
    displacement_y = cv2.resize(coarse_y, (256, 256), interpolation=cv2.INTER_CUBIC)

    result = []
    for mask in masks:
        transformed = cv2.warpAffine(mask, matrix, (256, 256), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        transformed = cv2.remap(transformed, grid_x + displacement_x, grid_y + displacement_y,
                                interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        # Very small source strokes can disappear under nearest-neighbor elastic
        # remapping. Fall back to the shared affine transform, preserving a real
        # supervised stroke instead of emitting an empty target.
        if not np.any(transformed):
            transformed = cv2.warpAffine(mask, matrix, (256, 256), flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kernel_size = int(rng.choice([1, 3, 3, 5, 5, 7, 9]))
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
            transformed = cv2.dilate(transformed, kernel)
        result.append((transformed > 0).astype(np.uint8))
    result = np.asarray(result, np.uint8)
    require(np.all(result.reshape(len(result), -1).sum(1) > 0),
            'augmentation produced empty stroke: ' + character)
    return result


def save_pair(destination, character, reference, target_masks, augmentation):
    with np.load(reference['reference_path'], allow_pickle=False) as z:
        reference_masks = z['reference_single_image'].astype(np.uint8)
        labels = z['stroke_label'].astype(np.int64)
        centroids = z['reference_single_centroid'].astype(np.float32)
        color = z['reference_color_image'].astype(np.float32)
    require(target_masks.shape == reference_masks.shape, 'reference/target shape mismatch')
    require(len(labels) == len(target_masks), 'label count mismatch')
    target = np.any(target_masks, axis=0).astype(np.uint8)[None]
    np.savez_compressed(destination,
        name=np.array(character),
        stroke_name=np.array(','.join(LABEL_NAMES[int(label)] for label in labels)),
        stroke_label=labels,
        reference_color_image=color,
        reference_single_image=reference_masks.astype(bool),
        reference_single_centroid=centroids,
        target_image=target.astype(bool),
        target_single_image=target_masks.astype(bool),
        reference_source=np.array('Make Me A Hanzi + CCSE-v2 mapped labels'),
        target_source=np.array('CCSE Kaiti COCO instances'),
        augmentation=np.array(augmentation))


def load_split(root, split):
    path = root / 'annotations' / ('instances_%s2021.json' % split)
    data = json.loads(path.read_text(encoding='utf-8'))
    annotations = {}
    for item in data['annotations']:
        annotations.setdefault(item['image_id'], []).append(item)
    return data['images'], annotations, digest(path)


def prepare_character(image, annotations, sequence, reference):
    character = image['font']
    require(sequence['category_status'] == 'complete', 'incomplete labels')
    require(sequence['stroke_count'] <= 30, 'stroke count exceeds model limit')
    items = annotations.get(image['id'], [])
    categories = [int(item['category_id']) for item in items]
    require(categories == sequence['raw_partial_stroke_labels'],
            'COCO order differs from sequence order: ' + character)
    require(len(items) == reference['stroke_count'] == sequence['stroke_count'],
            'stroke count mismatch: ' + character)
    masks = np.asarray([resize_mask(decode_annotation(item, image['height'], image['width']))
                        for item in items], np.uint8)
    require(np.all(masks.reshape(len(masks), -1).sum(1) > 0), 'empty COCO mask: ' + character)
    return masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ccse', required=True)
    p.add_argument('--sequences', required=True)
    p.add_argument('--references', required=True)
    p.add_argument('--rhsedb', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    require(a.workers > 0, 'workers must be positive')
    ccse, rhsedb = Path(a.ccse).resolve(), Path(a.rhsedb).resolve()
    sequences = {row['character']: row for row in records(a.sequences)}
    references = {row['character']: row for row in records(a.references)}
    out = output_dir(a.output)
    for name in ('train', 'test', 'heldout_test'):
        (out / name).mkdir()
    reports, errors, source_hashes = [], [], {}

    split_data = {}
    for split in ('train', 'val', 'test'):
        images, annotations, annotation_hash = load_split(ccse, split)
        source_hashes['ccse_' + split] = annotation_hash
        eligible = [image for image in images
                    if image['font'] in sequences and image['font'] in references
                    and sequences[image['font']]['category_status'] == 'complete'
                    and sequences[image['font']]['stroke_count'] <= 30
                    and all(annotation_has_mask(item, image['height'], image['width'])
                            for item in annotations.get(image['id'], []))]
        split_data[split] = (eligible, annotations)

    train_images, train_annotations = split_data['train']
    original_count = len(train_images)
    augmented_count = round(2.5 * original_count)
    rhsedb_count = round(1.5 * original_count)
    require(rhsedb_count <= len(list((rhsedb / 'train').glob('*.npz'))), 'insufficient RHSEDB')

    def build_original(image, destination_split, annotations, kind='original', variant=0):
        character = image['font']
        try:
            masks = prepare_character(image, annotations, sequences[character], references[character])
            if kind == 'augmented':
                masks = augment_masks(masks, character, variant)
            stem = ('ccse_%s_u%05X_v%02d.npz' % (kind, ord(character), variant))
            destination = out / destination_split / stem
            save_pair(destination, character, references[character], masks,
                      'none' if kind == 'original' else 'affine+elastic+per-stroke-dilation-v1')
            return dict(split=destination_split, kind=kind, character=character,
                        path=str(destination.resolve()), strokes=len(masks), variant=variant)
        except Exception as exc:
            return dict(error=str(exc), character=character, split=destination_split, kind=kind)

    jobs = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for image in train_images:
            jobs.append(pool.submit(build_original, image, 'train', train_annotations))
        for index in range(augmented_count):
            image = train_images[index % original_count]
            variant = index // original_count + 1
            jobs.append(pool.submit(build_original, image, 'train', train_annotations,
                                    'augmented', variant))
        for source_split, destination_split in [('val', 'test'), ('test', 'heldout_test')]:
            images, annotations = split_data[source_split]
            for image in images:
                jobs.append(pool.submit(build_original, image, destination_split, annotations))
        for future in jobs:
            item = future.result()
            (errors if 'error' in item else reports).append(item)

    rhsedb_paths = sorted((rhsedb / 'train').glob('*.npz'), key=lambda path: int(path.stem))
    rng = np.random.default_rng(seed_for('rhsedb-sample-v1', original_count))
    selected = sorted(rng.choice(len(rhsedb_paths), size=rhsedb_count, replace=False).tolist())
    for index in selected:
        source = rhsedb_paths[index]
        destination = out / 'train' / ('rhsedb_%06d.npz' % int(source.stem))
        os.symlink(source, destination)
        reports.append(dict(split='train', kind='rhsedb', character=None,
                            path=str(destination.resolve()), source_path=str(source.resolve())))

    counts = Counter((row['split'], row['kind']) for row in reports)
    train_total = sum(value for (split, _), value in counts.items() if split == 'train')
    ratios = {kind: counts[('train', kind)] / train_total
              for kind in ('rhsedb', 'original', 'augmented')}
    write_records(out / 'manifest.jsonl', reports)
    write_records(out / 'errors.jsonl', errors)
    write_json(out / 'summary.json', dict(
        schema='strokeextraction-mixed-sdnet-v1', train_total=train_total,
        counts={'%s/%s' % key: value for key, value in sorted(counts.items())},
        train_ratios=ratios, errors=len(errors), seed_policy='sha256 deterministic',
        augmentation=dict(rotation_degrees=10, scale=[.88, 1.12], translation_pixels=12,
                          elastic_coarse_grid=9, elastic_sigma_pixels=3,
                          dilation_kernel_choices=[1, 3, 3, 5, 5, 7, 9]),
        source_sha256=dict(source_hashes, sequences=digest(a.sequences),
                           references=digest(a.references)),
        split_policy='CCSE train -> train; CCSE val -> loader test; CCSE test -> heldout_test; RHSEDB sampled from train',
        ordering_evidence='All 9523 COCO annotation category sequences exactly match CCSE-v2 raw category sequences.'))
    require(not errors, 'dataset build produced errors; see errors.jsonl')
    print(json.dumps(dict(train_total=train_total, ratios=ratios,
                          test=counts[('test', 'original')],
                          heldout_test=counts[('heldout_test', 'original')]), indent=2))


if __name__ == '__main__':
    main()
