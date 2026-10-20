"""Package the complete CHTR source tree plus selected inference weights and LMDB datasets."""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        return chunk


def verify(archive, records):
    expected = {row['path']: row for row in records}
    seen = set()
    with gzip.open(archive, 'rb') as compressed:
        with tarfile.open(fileobj=compressed, mode='r|') as container:
            for member in container:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise RuntimeError('Unexpected archive member: ' + member.name)
                row = expected[member.name]
                digest = hashlib.sha256()
                with container.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if member.size != row['bytes'] or digest.hexdigest() != row['sha256']:
                    raise RuntimeError('Archive verification failed: ' + member.name)
                seen.add(member.name)
        # Consume the gzip trailer as well, validating CRC and the stream length.
        while compressed.read(1024 * 1024):
            pass
    if seen != set(expected):
        raise RuntimeError('Missing archive members')


def package(output, name, paths, threads):
    archive = output/name
    temporary = output/(name + '.partial')
    records = []
    process = None
    try:
        with temporary.open('xb') as destination:
            process = subprocess.Popen(['pigz', '-1', '-n', '-p', str(threads)],
                                       stdin=subprocess.PIPE, stdout=destination)
            with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as container:
                for path in paths:
                    before = identity(path)
                    member_name = 'CHTR/' + path.relative_to(PROJECT).as_posix()
                    info = container.gettarinfo(str(path), arcname=member_name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        reader = HashReader(stream)
                        container.addfile(info, reader)
                    if identity(path) != before:
                        raise RuntimeError('Source changed during packaging: ' + str(path))
                    records.append(dict(path=member_name, bytes=info.size, sha256=reader.digest.hexdigest()))
                    print('[file] ' + member_name, flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('pigz compression failed')
        print('[verify] ' + name, flush=True)
        verify(temporary, records)
        temporary.rename(archive)
        digest = sha256(archive)
        (output/(name + '.sha256')).write_text(digest + '  ' + name + '\n')
        result = dict(archive=name, bytes=archive.stat().st_size, sha256=digest,
                      source_bytes=sum(row['bytes'] for row in records), verified=True, files=records)
        print('[complete] ' + name + ' bytes=' + str(result['bytes']), flush=True)
        return result
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        temporary.unlink(missing_ok=True)
        raise


def select(datasets):
    # Include tracked and untracked source files, honoring the project's asset ignores.
    raw = subprocess.check_output(['git', '-C', str(PROJECT), 'ls-files',
                                   '--cached', '--others', '--exclude-standard', '-z'])
    paths = {PROJECT/name for name in raw.decode().split('\0') if name}
    paths = {path for path in paths if path.is_file()}
    job = PROJECT/'models/runs/full_separated_10gb/job_237'
    names = ['best.pt', 'last.pt', 'config.json', 'stroke_type_map.json',
             'training_status.json', 'dataset_selection.json', 'epoch_metrics.jsonl', 'metrics.jsonl']
    paths.update(job/name for name in names)
    extract = PROJECT/'third_party/StrokeExtraction/out'
    paths.update([extract/'SDNet_MMAH_CCSEDB_v4/model/sdnet_model.pth',
                  extract/'SegNet_MMAH_CCSEDB_v4/model/model.pth',
                  extract/'ExtractNet_MMAH_CCSEDB_v4/model/model_extract.pth'])
    vs = PROJECT/'third_party/virtual_sketching/outputs/snapshot/pretrain_clean_line_drawings'
    paths.update(vs/name for name in ['checkpoint', 'model_config.json',
                                    'p2s-75000.index', 'p2s-75000.data-00000-of-00001'])
    for key, folder in [('full', 'chtr_mmah_lmdb'), ('1915', 'chtr_mmah_lmdb_1915')]:
        if datasets in ('both', key):
            # Omit machine-local LMDB lock files; all sample assets live in data.mdb.
            paths.update(PROJECT/'data'/folder/name for name in
                         ['data.mdb', 'summary.json', 'complete.json', 'stroke_type_map.json'])
    return [('CHTR_epoch100_linux.tar.gz', sorted(paths))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'dist/epoch100_bundle')
    parser.add_argument('--datasets', choices=('both', 'full', '1915'), default='both')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    groups = select(args.datasets)
    for _, paths in groups:
        for path in paths:
            if not path.is_file() or path.is_symlink():
                raise SystemExit('Missing regular source file: ' + str(path))
    for name, paths in groups:
        print('[plan] ' + name + ' files=' + str(len(paths)) +
              ' bytes=' + str(sum(p.stat().st_size for p in paths)), flush=True)
    if args.dry_run:
        return
    if shutil.which('pigz') is None:
        raise SystemExit('pigz is required for parallel gzip compression')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Coordinate with dataset publication without modifying either LMDB directory.
    with (PROJECT/'data/.chtr_mmah_lmdb.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        results = []
        for name, paths in groups:
            results.append(package(output, name, paths, args.threads))
            (output/'manifest.json').write_text(json.dumps(dict(status='in_progress', packages=results),
                                                          ensure_ascii=False, indent=2)+'\n')
        source_guide = PROJECT/'reproduction/LOCAL_EPOCH100.md'
        shutil.copyfile(source_guide, output/'README_LOCAL.md')
        (output/'SHA256SUMS').write_text(''.join(row['sha256'] + '  ' + row['archive'] + '\n' for row in results))
        (output/'manifest.json').write_text(json.dumps(dict(status='complete', datasets=args.datasets,
            packages=results, note='Complete source with relative CHTR paths; selected inference weights and datasets; no environments or training data for third-party models.'),
            ensure_ascii=False, indent=2)+'\n')
    print('[done] ' + str(output), flush=True)


if __name__ == '__main__':
    main()
