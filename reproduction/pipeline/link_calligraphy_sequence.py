"""Link calligraphy images to same-character ordered Kai candidates for review."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import digest, output_dir, read_json, records, require, write_json, write_records


def make_links(calligraphy, sequences):
    by_character = {}
    for row in sequences:
        character = row['character']
        require(len(character) == 1 and character not in by_character, 'duplicate/invalid sequence character')
        require(row['category_system'] == 'CCSE-25', 'unexpected category system')
        require(row['category_status'] in ('complete', 'unknown_due_to_upstream_missing_annotations'), 'unexpected category status')
        require(row['strokeextraction_24_compatible'] is False, 'sequence claims unverified compatibility')
        by_character[character] = row
    seen = set()
    linked = []
    counts = Counter()
    characters = set()
    for row in calligraphy:
        require(len(row['character']) == 1 and row['sample_id'] not in seen, 'duplicate/invalid calligraphy sample')
        require(row['pack_eligible'] is False and row['provenance_status'] == 'writer_and_style_unknown', 'unexpected provenance')
        seen.add(row['sample_id'])
        character = row['character']
        characters.add(character)
        candidate = by_character.get(character)
        status = 'no_same_character_template' if candidate is None else ('candidate_labels_incomplete' if candidate['category_status'] != 'complete' else 'candidate_review_required')
        output = dict(schema=row['schema'], sample_id=row['sample_id'], character=character,
                      image_path=row['image_path'], source_image_path=row['source_image_path'],
                      source_sha256=row['source_sha256'], provenance_status=row['provenance_status'],
                      pack_eligible=False, match_basis='unicode_character_only',
                      match_status=status, registration_status='not_evaluated',
                      review_status='pending', strokeextraction_24_compatible=False)
        if candidate:
            output.update(template_sample_id=candidate['sample_id'],
                          ordered_strokes_path=candidate['ordered_strokes_path'],
                          ordered_strokes_sha256=candidate['ordered_strokes_sha256'],
                          template_preview_path=candidate['preview_path'],
                          ccse_image_path=candidate['ccse_image_path'],
                          ccse_split=candidate['ccse_split'], category_system='CCSE-25',
                          category_status=candidate['category_status'],
                          stroke_count=candidate['stroke_count'])
        linked.append(output)
        counts[status] += 1
    return linked, dict(images=len(linked), calligraphy_characters=len(characters),
                        matched_characters=len(characters & by_character.keys()),
                        unmatched_calligraphy_characters=len(characters - by_character.keys()),
                        templates_without_calligraphy=len(by_character.keys() - characters),
                        template_characters_with_incomplete_labels=sum(by_character[c]['category_status'] != 'complete' for c in characters & by_character.keys()),
                        by_status=dict(counts))


def diagnostic_iou(image_path, preview_path):
    """Direct raster overlap only; no registration or stroke alignment."""
    a = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(preview_path), cv2.IMREAD_GRAYSCALE)
    require(a is not None and b is not None, 'missing diagnostic image')
    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_NEAREST) < 180
    b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_NEAREST) < 180
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calligraphy', required=True, help='Normalized calligraphy manifest')
    parser.add_argument('--sequences', required=True, help='Ordered CCSE-Kai sequence manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-samples', type=int, default=0, help='Number of deterministic matched samples for direct overlap diagnostics')
    args = parser.parse_args()
    require(args.audit_samples >= 0, 'audit-samples must be nonnegative')
    calligraphy_path, sequence_path = Path(args.calligraphy).resolve(), Path(args.sequences).resolve()
    sequences = records(sequence_path)
    require(read_json(sequence_path.parent / 'category_map.json')['category_system'] == 'CCSE-25', 'category map mismatch')
    linked, summary = make_links(records(calligraphy_path), sequences)
    summary.update(calligraphy_manifest_sha256=digest(calligraphy_path),
                   sequence_manifest_sha256=digest(sequence_path),
                   category_map_sha256=digest(sequence_path.parent / 'category_map.json'),
                   caveat='Same Unicode character is only a candidate match. Geometry and style require review; CCSE-25 is not StrokeExtraction-24.')
    output = output_dir(args.output)
    write_records(output / 'manifest.jsonl', linked)
    if args.audit_samples:
        candidates = [row for row in linked if row['match_status'] != 'no_same_character_template']
        stride = max(1, len(candidates) // args.audit_samples)
        selected = candidates[::stride][:args.audit_samples]
        diagnostics = []
        for row in selected:
            diagnostics.append(dict(sample_id=row['sample_id'], character=row['character'],
                calligraphy_vs_template_iou=diagnostic_iou(row['image_path'], row['template_preview_path']),
                ccse_vs_template_iou=diagnostic_iou(row['ccse_image_path'], row['template_preview_path']),
                interpretation='Unregistered direct overlap; diagnostic only.'))
        write_records(output / 'overlap_diagnostics.jsonl', diagnostics)
        summary['overlap_diagnostic_samples'] = len(diagnostics)
    write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
