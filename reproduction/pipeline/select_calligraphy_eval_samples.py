"""Select deterministic calligraphy samples by character and attach reference-bank entries."""
import argparse
from pathlib import Path

from common import output_dir, records, require, write_json, write_records


def select_evenly(rows, count):
    require(rows and count > 0, 'empty rows or invalid count')
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[index] for index in indices]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calligraphy', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--characters', nargs='+', required=True)
    p.add_argument('--per-character', type=int, default=2)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    require(a.per_character > 0, 'per-character must be positive')
    require(len(set(a.characters)) == len(a.characters), 'duplicate requested character')
    bank = {row['character']: row for row in records(a.bank)}
    grouped = {character: [] for character in a.characters}
    for row in records(a.calligraphy):
        if row['character'] in grouped:
            grouped[row['character']].append(row)
    samples, references, summary = [], [], []
    for character in a.characters:
        require(character in bank, 'character absent from bank: ' + character)
        rows = sorted(grouped[character], key=lambda row: row['sample_id'])
        require(rows, 'character absent from calligraphy manifest: ' + character)
        chosen = select_evenly(rows, a.per_character)
        for row in chosen:
            item = dict(row)
            item.update(reference_source='Make Me A Hanzi + CCSE-v2 mapped labels',
                        evaluation_stroke_count=bank[character]['stroke_count'])
            samples.append(item)
        references.append(bank[character])
        summary.append(dict(character=character, stroke_count=bank[character]['stroke_count'],
                            available_images=len(rows), selected=[row['sample_id'] for row in chosen]))
    out = output_dir(a.output)
    write_records(out / 'samples.jsonl', samples)
    write_records(out / 'references.jsonl', references)
    write_json(out / 'summary.json', dict(samples=len(samples), characters=len(a.characters),
               per_character=a.per_character, selection='evenly spaced in sorted sample_id order',
               details=summary))


if __name__ == '__main__':
    main()
