"""Fail-closed Docker execution for the frozen MovieLens training worker.

Only allowlisted code, train files and unlabeled coordinates are mounted.
A real Docker runtime is required for OS-isolation claims.
"""
import argparse
import json
import re
import shutil
import tempfile
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVISED = ROOT / 'revised'
STAGE_FILES = ('isolated_runner_37.py', 'isolation_36.py', 'runner_33.py',
               'runner_30.py', 'validation_32.py', 'bias_baseline_29.py',
               'benchmark_plan_28.json')
REVISED_FILES = ('sparse_svp.py', 'scalable_19.py', 'resource_guard_20.py')
PINNED_IMAGE = re.compile(r'^[a-zA-Z0-9._/-]+(?:\:[0-9]+)?@sha256:[0-9a-f]{64}$')


def stage_reviewed_code(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'prospective').mkdir()
    (destination / 'revised').mkdir()
    for filename in STAGE_FILES:
        shutil.copyfile(HERE / filename, destination / 'prospective' / filename)
    for filename in REVISED_FILES:
        shutil.copyfile(REVISED / filename, destination / 'revised' / filename)
    return destination


def docker_command(family, params, fit_dir, coordinates, *, stage_dir, image, seconds=120):
    if family not in {'global_mean', 'user_item_bias', 'SVP', 'SVT', 'ADMiRA_lab_variant'}:
        raise ValueError('unknown model family')
    if not isinstance(params, dict) or not 1 <= seconds <= 120:
        raise ValueError('invalid params or timeout')
    if not PINNED_IMAGE.fullmatch(image):
        raise ValueError('image must be pinned to a real-looking sha256 digest')
    stage = Path(stage_dir).resolve(strict=True)
    if not all((stage / 'prospective' / f).is_file() for f in STAGE_FILES):
        raise ValueError('incomplete prospective source staging')
    if not all((stage / 'revised' / f).is_file() for f in REVISED_FILES):
        raise ValueError('incomplete revised source staging')
    fit = Path(fit_dir).resolve(strict=True)
    coords = Path(coordinates).resolve(strict=True)
    if not fit.is_dir() or not coords.is_file():
        raise ValueError('invalid fit or coordinate file')
    if {p.name for p in fit.iterdir()} != {'train.csv', 'catalog.json'}:
        raise ValueError('fit directory must contain exactly train.csv and catalog.json')
    if any(not (fit / filename).is_file() for filename in ('train.csv', 'catalog.json')):
        raise ValueError('missing training file')
    # Reject input paths nested in each other, preventing unintended additional mounts.
    if coords.is_relative_to(fit) or fit.is_relative_to(coords.parent) and coords.name in {'train.csv','catalog.json'}:
        raise ValueError('coordinates must be separate from training directory')
    return [
        'docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--read-only',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--pids-limit', '128', '--memory', '2g', '--memory-swap', '2g',
        '--cpus', '1', '--user', '65534:65534',
        '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=64m',
        '--mount', f'type=bind,src={fit},dst=/input/fit,readonly',
        '--mount', f'type=bind,src={coords},dst=/input/coordinates.npz,readonly',
        '--mount', f'type=bind,src={stage},dst=/code,readonly',
        '--workdir', '/code/prospective',
        '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'OMP_NUM_THREADS=1',
        '--env', 'MKL_NUM_THREADS=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        image, 'python', '/code/prospective/isolated_runner_37.py', '--worker',
        family, json.dumps(params, sort_keys=True), '/input/fit', '/input/coordinates.npz'
    ]


def execute(family, params, fit_dir, coordinates, *, image, seconds=120):
    if not shutil.which('docker'):
        raise RuntimeError('Docker unavailable: refusing unisolated fallback')
    with tempfile.TemporaryDirectory(prefix='reviewed_training_code_') as temp:
        # TemporaryDirectory is mode 0700 by default: uid 65534 in Docker
        # cannot traverse it. Only allowlisted code is placed beneath this path.
        Path(temp).chmod(0o755)
        stage = stage_reviewed_code(Path(temp) / 'source')
        command = docker_command(family, params, fit_dir, coordinates, stage_dir=stage, image=image, seconds=seconds)
        result = subprocess.run(command, text=True, capture_output=True, timeout=seconds+15)
    if result.returncode:
        raise RuntimeError(f'Container failed (exit={result.returncode}): {result.stderr[-1500:]}')
    payload = json.loads(result.stdout)
    if payload.get('status') != 'ok':
        raise RuntimeError('Worker did not report success')
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('family'); p.add_argument('params_json'); p.add_argument('fit_dir'); p.add_argument('coordinates')
    p.add_argument('--image', required=True, help='Image reference pinned to a verified sha256 digest')
    a = p.parse_args()
    print(json.dumps(execute(a.family, json.loads(a.params_json), a.fit_dir, a.coordinates, image=a.image)))
