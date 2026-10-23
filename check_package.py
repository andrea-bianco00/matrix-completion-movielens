"""Run offline package checks without MovieLens labels or network access.

This is a convenience entry point, not a claim of clean-room reproducibility.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_checks(*, python=sys.executable):
    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    commands = [
        ('release_integrity', [python, 'verify_release.py']),
        ('algorithm_tests', [python, '-m', 'unittest', 'discover', '-s', 'revised', '-p', 'test*.py', '-q']),
        ('prospective_tests', [python, '-m', 'unittest', 'discover', '-s', 'prospective', '-p', 'test*.py', '-q']),
        ('release_tests', [python, '-m', 'unittest', 'test_verify_release', '-q']),
    ]
    results = []
    for name, command in commands:
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        results.append({'check': name, 'exit_code': completed.returncode,
                        'stdout_tail': completed.stdout[-1200:], 'stderr_tail': completed.stderr[-1200:]})
        if completed.returncode:
            break
    return {'status': 'passed' if len(results) == len(commands) and all(r['exit_code'] == 0 for r in results) else 'failed',
            'python': sys.version.split()[0], 'checks': results,
            'scope': 'offline artifact verification and automated tests only; not a fresh dependency install or Docker isolation test'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional path for JSON results')
    args = parser.parse_args()
    report = run_checks()
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    sys.exit(0 if report['status'] == 'passed' else 1)
