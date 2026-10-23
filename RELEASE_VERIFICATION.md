# Release verification — 23 October 2026

## Commands executed

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s revised -p 'test*.py' -q` — 85 passed.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s prospective -p 'test*.py' -q` — 55 passed.
- `python -m unittest test_verify_release -q` — 3 passed.
- `python verify_release.py` — verified, winner `user_item_bias`, 15,126 test ratings, RMSE 0.8780574841321015.

## Scope

The new offline verifier validates frozen artifact hashes, selection consistency and decomposition of final test RMSE into warm/cold groups. Tampering tests reject changed winner parameters or a changed test RMSE. This is **not** an independent rerun of training or test scoring and does not certify OS-level sandbox isolation. No held-out ratings were opened by these checks.

## Release blockers

Coauthor permission, license choice, path-independent reproduction workflow and clean-environment validation remain open. Docker runtime isolation has not been verified. GitHub and portfolio have not been changed.
