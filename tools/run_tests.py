"""Run original assertion-based test programs plus maintained regressions."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    tests = sorted(ROOT.glob('test_exact_*.py'))
    tests += sorted((ROOT / 'research/tests').glob('test_*.py'))
    tests += sorted((ROOT / 'research/ev').glob('test_*.py'))
    tests += sorted((ROOT / 'tests').glob('test_*.py'))
    failed = []
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    for path in tests:
        result = subprocess.run([sys.executable, '-B', str(path)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        label = path.relative_to(ROOT)
        print(f'{"PASS" if result.returncode == 0 else "FAIL"} {label}')
        if result.returncode:
            failed.append(str(label))
            print(result.stdout + result.stderr)
    print(f'{len(tests) - len(failed)}/{len(tests)} test programs passed')
    return bool(failed)

if __name__ == '__main__':
    sys.exit(main())
