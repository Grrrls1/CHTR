"""Download only an official OLHWDB POT test archive; retain archive and resume partial downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

BASE = 'https://nlpr.ia.ac.cn/databases/Download/Online/CharData/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['1.0', '1.1', '1.2'], default='1.1')
    parser.add_argument('--output', type=Path, default=Path('data/CASIA-OLHWDB'))
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root/f'Pot{args.version}Test.zip'
    url = BASE+archive.name
    if not archive.exists():
        if args.extract_only:
            parser.error('Archive missing: '+str(archive))
        partial = archive.with_suffix('.zip.part')
        subprocess.run(['wget', '--continue', '--tries=8', '--timeout=30', '--output-document', str(partial), url], check=True)
        with zipfile.ZipFile(partial) as z:
            bad = z.testzip()
            if bad:
                raise ValueError('ZIP CRC mismatch: '+bad)
        partial.replace(archive)
    target = root/args.version/'test'
    target.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError('ZIP CRC mismatch: '+bad)
        required = sum(i.file_size for i in z.infolist() if not i.is_dir())
        if required > shutil.disk_usage(target).free:
            raise OSError('Insufficient free space for extraction')
        for item in z.infolist():
            path = target/item.filename
            if target.resolve() not in path.resolve().parents or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Unsafe ZIP path: '+item.filename)
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            data = z.read(item)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
                    raise ValueError('Existing file differs; refusing overwrite: '+str(path))
                skipped += 1
                continue
            fd, temporary = tempfile.mkstemp(prefix='.pot-', dir=path.parent)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(data)
                os.link(temporary, path)
                written += 1
            finally:
                Path(temporary).unlink(missing_ok=True)
    report = dict(url=url, archive=str(archive), bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  extraction=str(target), extracted=written, existing_verified=skipped,
                  pot_files=len(list(target.rglob('*.pot'))), split='official test')
    (root/f'download_{args.version}_test.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
