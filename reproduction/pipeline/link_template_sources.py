"""Create a source-aware calligraphy candidate index across three template datasets."""
import argparse
from collections import Counter
from pathlib import Path

from common import digest, output_dir, read_json, records, require, write_json, write_records


def index(rows, kind):
    result = {}
    for row in rows:
        char = row['character']
        require(len(char) == 1 and char not in result, 'duplicate/invalid %s character' % kind)
        result[char] = row
    return result


def candidates_for(character, ccse, mmah, rhsedb, shared_geometry=frozenset()):
    candidates = []
    if character in ccse:
        r = ccse[character]
        require(r['category_system'] == 'CCSE-25' and r['strokeextraction_24_compatible'] is False, 'unsafe CCSE category metadata')
        candidates.append(dict(source_id='lee_xeo_ccse_kaiti', template_path=r['ordered_strokes_path'],
                               template_sha256=r['ordered_strokes_sha256'], preview_path=r['preview_path'],
                               stroke_count=r['stroke_count'], category_system='CCSE-25',
                               category_status=r['category_status'],
                               strokeextraction_24_compatible=False, registration_status='not_evaluated',
                               order_source='Make Me A Hanzi via Lee-xeo'))
    if character in mmah:
        r = mmah[character]
        require(r['category_system'] == 'unlabeled' and r['strokeextraction_24_compatible'] is False, 'unsafe Make Me A Hanzi category metadata')
        candidates.append(dict(source_id='makemeahanzi_official', template_path=r['ordered_strokes_path'],
                               template_sha256=r['ordered_strokes_sha256'], preview_path=r['preview_path'],
                               stroke_count=r['stroke_count'], category_system='unlabeled',
                               category_status='no_stroke_categories_in_source',
                               strokeextraction_24_compatible=False, registration_status='not_evaluated',
                               order_source='Make Me A Hanzi',
                               same_geometry_as_lee_xeo=character in shared_geometry))
    if character in rhsedb:
        r = rhsedb[character]
        require(r['category_system'] == 'StrokeExtraction-24', 'unsafe RHSEDB category metadata')
        candidates.append(dict(source_id='rhsedb_reference_bank', template_path=r['reference_path'],
                               template_sha256=r['reference_sha256'], stroke_count=r['stroke_count'],
                               category_system='StrokeExtraction-24', category_status='complete',
                               strokeextraction_24_compatible=True, registration_status='not_evaluated',
                               source_split=r['source_split'], order_source='RHSEDB'))
    return candidates


def make_links(calligraphy, ccse, mmah, rhsedb, shared_geometry=frozenset()):
    ccse, mmah, rhsedb = index(ccse, 'CCSE'), index(mmah, 'Make Me A Hanzi'), index(rhsedb, 'RHSEDB')
    seen, chars, counts = set(), set(), Counter()
    rows = []
    for r in calligraphy:
        require(len(r['character']) == 1 and r['sample_id'] not in seen, 'duplicate/invalid calligraphy sample')
        require(r['pack_eligible'] is False and r['provenance_status'] == 'writer_and_style_unknown', 'unexpected calligraphy provenance')
        seen.add(r['sample_id'])
        chars.add(r['character'])
        candidates = candidates_for(r['character'], ccse, mmah, rhsedb, shared_geometry)
        for c in candidates:
            counts[c['source_id'] + '_images'] += 1
        if not candidates:
            counts['no_template_images'] += 1
        rows.append(dict(schema=r['schema'], sample_id=r['sample_id'], character=r['character'],
                         image_path=r['image_path'], source_image_path=r['source_image_path'],
                         source_sha256=r['source_sha256'], provenance_status=r['provenance_status'],
                         pack_eligible=False, match_basis='unicode_character_only',
                         review_status='pending', template_candidates=candidates))
    summary = dict(images=len(rows), calligraphy_characters=len(chars),
                   no_template_characters=len(chars - (ccse.keys() | mmah.keys() | rhsedb.keys())),
                   sources={name:dict(total_characters=len(data), matched_characters=len(chars & data.keys()),
                                      matched_images=counts[name + '_images'])
                            for name, data in [('lee_xeo_ccse_kaiti', ccse),
                                               ('makemeahanzi_official', mmah),
                                               ('rhsedb_reference_bank', rhsedb)]},
                   no_template_images=counts['no_template_images'],
                   caveat='Same-character candidates only. Make Me A Hanzi and Lee-xeo share upstream strokes; neither supplies verified StrokeExtraction-24 labels. All registration/review pending.')
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('calligraphy', 'ccse', 'makemeahanzi', 'rhsedb', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--source-comparison', required=True, help='Verified official/Lee-xeo geometry comparison report')
    args = p.parse_args()
    paths = {name: Path(getattr(args, name)).resolve() for name in ('calligraphy', 'ccse', 'makemeahanzi', 'rhsedb')}
    source_rows = {name: records(path) for name, path in paths.items()}
    comparison = read_json(args.source_comparison)
    ccse_chars = {r['character'] for r in source_rows['ccse']}
    mmah_chars = {r['character'] for r in source_rows['makemeahanzi']}
    shared = ccse_chars & mmah_chars
    require(comparison['shared_characters'] == len(shared) and
            comparison['identical_stroke_paths'] == len(shared) and
            comparison['identical_medians'] == len(shared), 'shared geometry comparison incomplete')
    rows, summary = make_links(*(source_rows[name] for name in paths), shared_geometry=shared)
    summary['shared_geometry_characters'] = len(shared)
    summary['source_comparison_sha256'] = digest(args.source_comparison)
    summary['input_manifest_sha256'] = {name: digest(path) for name, path in paths.items()}
    output = output_dir(args.output)
    write_records(output / 'manifest.jsonl', rows)
    write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
