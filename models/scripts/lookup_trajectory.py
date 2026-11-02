"""Look up text in an exported trajectory dictionary, without GPU or model dependencies."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.trajectory_dictionary import TrajectoryDictionary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text')
    group.add_argument('--list', action='store_true')
    parser.add_argument('--pixels', action='store_true', help='Default exports normalized XYW')
    parser.add_argument('--output', type=Path, help='Optional new JSON file; default stdout')
    args = parser.parse_args()
    library = TrajectoryDictionary(args.library)
    try:
        result = dict(available_characters=library.available_characters()) if args.list else library.lookup_text(args.text, normalized=not args.pixels)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    value = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(value)
        print(str(args.output.resolve()))
    else:
        print(value, end='')


if __name__ == '__main__':
    main()
