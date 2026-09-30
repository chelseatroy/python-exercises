"""Copies the exercise files students download from the course site into docs/downloads/.

Run from anywhere after changing any of the files or folders below, then commit docs/downloads/:

    python3 tools/build_downloads.py

`--check` reports whether docs/downloads/ is out of date without changing it (exit code 1 if it is).

Folders become zip files. Only files tracked by git are included, minus Jupyter checkpoints and
bytecode caches. Zips are written with fixed timestamps, so rebuilding unchanged files changes nothing.
"""
import io
import os
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'downloads')

# Download name: source path in the repo. A folder becomes <name>.zip with the folder at its top level.
DOWNLOADS = {
    'table.py': 'homework-2/table.py',
    'test_table.py': 'homework-2/test_table.py',
    'test_framework_exercise.zip': 'test_framework_exercise',
    'phoenix_test.zip': 'test_framework_exercise/phoenix_test',
    'phoenixcel.zip': 'data_frame_exercise/phoenixcel',
    'language_generator_exercise.zip': 'language_generator_exercise',
    's8_malloc_challenge.ipynb': 's8_malloc_challenge.ipynb',
    's8_concurrency_challenge.py': 's8_concurrency_challenge.py',
    's9_code_quality_exercise.zip': 's9_code_quality_exercise',
}

SKIP_PARTS = {'.ipynb_checkpoints', '__pycache__', '.DS_Store'}


def tracked_files(folder):
    out = subprocess.run(['git', 'ls-files', '-z', folder], cwd=REPO, capture_output=True, check=True).stdout
    paths = [p.decode() for p in out.split(b'\0') if p]
    return sorted(p for p in paths if not SKIP_PARTS & set(p.split('/')))


def zip_bytes(folder):
    top = os.path.basename(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in tracked_files(folder):
            info = zipfile.ZipInfo(top + '/' + os.path.relpath(path, folder), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(os.path.join(REPO, path), 'rb') as f:
                z.writestr(info, f.read())
    return buf.getvalue()


def build():
    files = {}
    for name, source in DOWNLOADS.items():
        if name.endswith('.zip'):
            files[name] = zip_bytes(source)
        else:
            with open(os.path.join(REPO, source), 'rb') as f:
                files[name] = f.read()
    return files


def main():
    check = '--check' in sys.argv
    files = build()
    existing = set(os.listdir(OUT)) if os.path.isdir(OUT) else set()
    stale = sorted(n for n, data in files.items()
                   if n not in existing or open(os.path.join(OUT, n), 'rb').read() != data)
    extra = sorted(existing - set(files))
    if check:
        for n in stale:
            print(f'out of date: docs/downloads/{n}')
        for n in extra:
            print(f'not in DOWNLOADS: docs/downloads/{n}')
        sys.exit(1 if stale or extra else 0)
    os.makedirs(OUT, exist_ok=True)
    for n in stale:
        with open(os.path.join(OUT, n), 'wb') as f:
            f.write(files[n])
        print(f'wrote docs/downloads/{n}')
    for n in extra:
        os.remove(os.path.join(OUT, n))
        print(f'removed docs/downloads/{n}')
    if not stale and not extra:
        print('docs/downloads/ is up to date')


if __name__ == '__main__':
    main()
