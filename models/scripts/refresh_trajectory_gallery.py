"""Refresh saved evaluation visualizations without model inference or LMDB reads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.visualization.trajectory_gallery import write_character_gallery, write_evaluation_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.evaluation_dir.resolve()
    summary = json.loads((output/'summary.json').read_text())
    selected = set(summary['visualization_ids'])
    rows = {sid: [] for sid in selected}
    with (output/'strokes.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['sample_id'] in rows:
                rows[row['sample_id']].append(row)
    titles = {}
    for sid in summary['visualization_ids']:
        if not rows[sid]:
            continue
        titles[sid] = rows[sid][0].get('character','')+' '+sid
        expected = len(rows[sid]) if summary['full_split'] else None
        report = write_character_gallery(output/sid, titles[sid], rows[sid], expected_strokes=expected)
        print(json.dumps(dict(sample_id=sid,**report),ensure_ascii=False),flush=True)
    write_evaluation_index(output,summary,titles)
    print(str(output/'index.html'),flush=True)


if __name__ == '__main__':
    main()
