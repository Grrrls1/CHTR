"""Prepare a small extraction evaluation from existing sample manifests and an MMAH bank."""
import argparse
from pathlib import Path

from common import output_dir, records, require, write_json, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variant', action='append', default=[],
                   help='Unicode character replacement SOURCE=TARGET, e.g. 迟=遲')
    a = p.parse_args()
    variants = {}
    for item in a.variant:
        source, separator, target = item.partition('=')
        require(separator and len(source) == 1 and len(target) == 1,
                'variant must be one Unicode character per side')
        variants[source] = target
    rows = []
    for path in a.samples:
        rows.extend(records(path))
    ids = [row['sample_id'] for row in rows]
    require(len(ids) == len(set(ids)), 'duplicate sample ID')
    bank = {row['character']: row for row in records(a.bank)}
    references = {}
    prepared = []
    for row in rows:
        source_character = row['character']
        character = variants.get(source_character, source_character)
        require(character in bank, 'missing bank character: ' + character)
        item = dict(row)
        item.update(character=character, source_character=source_character,
                    character_variant_status=('reassigned_by_visual_audit'
                                              if character != source_character else 'unchanged'),
                    reference_source='Make Me A Hanzi + CCSE v2 user-mapped')
        prepared.append(item)
        references[character] = bank[character]
    out = output_dir(a.output)
    write_records(out / 'samples.jsonl', prepared)
    write_records(out / 'references.jsonl', [references[key] for key in sorted(references)])
    write_json(out / 'summary.json', dict(samples=len(prepared), characters=sorted(references),
               variants=variants, note='Evaluation inputs only; variant reassignment is an explicit audit decision.'))


if __name__ == '__main__':
    main()
