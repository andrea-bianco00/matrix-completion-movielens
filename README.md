# Dimension Reduction & Matrix Completion — MovieLens

**Andrea Bianco and Matteo Mancini** · ENSAI university project · research/educational implementation.

> **Result, not a leaderboard claim.** On a predeclared 70/15/15 rating split of MovieLens Latest Small, a regularized user–item bias baseline outperformed the tested SVP, SVT and educational ADMiRA variant configurations on validation. The preselected baseline obtained **test RMSE 0.8780574841** (15,126 ratings). The held-out test has already been opened and must not be reused for tuning.

## What is studied

Low-rank matrix completion from sparse observed user–movie ratings, comparing **SVP** (rank-constrained projected gradient), **SVT** (nuclear-norm proximal gradient; not necessarily the original dual SVT algorithm), and an **ADMiRA-inspired educational variant** (not a claim of exact equivalence to the published ADMiRA algorithm). We also compare global-mean and regularized user–item bias baselines. The original ENSAI assignment motivated the implementations; this repository is the revised, audited research code.

## Experimental design

- Dataset: MovieLens **Latest Small** `ratings.csv` (100,836 ratings; 610 users and 9,724 rated movies). Dataset is **not redistributed**. Download the original `ml-latest-small.zip` from [GroupLens](https://grouplens.org/datasets/movielens/latest/).
- Split: 70,585 train / 15,125 validation / 15,126 test observations; frozen indices and SHA-256 checks in `prospective/split_manifest_18.json`. The split is **rating-wise**, not user-wise or chronological. The full set of user and movie IDs is assumed known.
- Hyperparameter grid and fixed solver budgets: `prospective/benchmark_plan_28.json`. Selection was by **unclipped validation RMSE**. Solver iteration limits were frequently reached; convergence is **not** claimed.
- Validation comparison: `prospective/validation_results_34.json`; pre-test selection lock: `prospective/selection_lock_41.json`. Final one-shot global winner result: `prospective/final_winner_results_43.json`.
- The earlier audit used the validation split once for a scoring-code check before the final grid comparison; exploratory work also used MovieLens with other splits. Therefore this is **not** an entirely independent confirmation dataset.

## Reported results

| Best configuration per family | Validation RMSE |
| --- | ---: |
| Regularized user–item bias (regularization 5) | **0.860652** |
| Global mean | 1.035442 |
| SVP (rank 5) | 1.38950 |
| ADMiRA-inspired lab variant (rank 2) | 1.49071 |
| Nuclear-norm proximal SVT (threshold 30) | 1.71912 |

Only the **preselected global winner** has a completed test evaluation: **raw test RMSE 0.8780574841**, warm-item RMSE **0.8713179204** (14,425 observations), cold-item RMSE **1.0067756094** (701 observations). Clipped test RMSE 0.8780037941 is **descriptive only**; clipping was not used for selection. No test performance is claimed for the other families.

The result supports only the **specific implementations, fixed hyperparameter grid, split, and computational budgets** evaluated here. It does not establish a universal ordering of matrix-completion methods.

## One-command offline verification

From the repository root, after installing the dependencies, run:

```bash
python check_package.py
```

This executes the offline release-integrity verifier, algorithm tests, prospective-protocol tests, and release-verifier tests. It does **not** need MovieLens labels or rerun the held-out test. A successful check does **not** certify a fresh installation or a real Docker sandbox.

## Running checks

Python with NumPy and SciPy is required. On Linux:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s revised -p 'test*.py' -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s prospective -p 'test*.py' -q
```

**Current package check:** all **85 algorithm tests and 55 prospective tests pass** in this environment (Python 3.13, one BLAS thread). The four formerly failing isolated-runner tests now construct their own **synthetic** training fixture, with no dependency on excluded MovieLens exports. This is a local verification, not proof of reproducibility on every operating system or Python version. Some historical experiment scripts still require the **original assignment archive** at audit-era locations; they are retained for traceability and are **not** a polished one-command benchmark. Exact experimental reproduction requires the original source ZIP matching the frozen checksum and is **not** a new untouched test evaluation.

## Data access and limitations

`prospective/isolation_36.py` exports disjoint partitions. `prospective/isolated_runner_37.py` trains using the `fit/` partition and unlabeled query coordinates. The Docker isolation launcher is **implemented but not validated by a real container run**; therefore OS-level denial of access to reserved files is **not certified**. Dataset labels, private exports and the original source archive are intentionally excluded from this package.

This repository contains a historical research/audit codebase rather than a production recommendation service. No license is asserted for the dataset; consult GroupLens terms. Andrea confirms that coauthor Matteo Mancini agrees to GitHub publication and the MIT software license. The code is released under [MIT](LICENSE); this does not license the MovieLens dataset. See [PUBLISHING.md](PUBLISHING.md).

## Verify the published artifacts without MovieLens data

Run the offline integrity check from the repository root:

```bash
python verify_release.py
python -m unittest test_verify_release -q
```

This verifies the frozen selection, recorded file hashes, final model identity and the mathematical consistency of aggregate/warm/cold test RMSE. It **does not** recompute predictions or certify that the historical test evaluation was independently reproduced. It requires no dataset and does not open reserved labels. The verifier uses the Python standard library only.

## Reproduce the selected model's training fit (without rescoring test)

Given the original assignment ZIP containing the unchanged `ml-latest-small/ratings.csv`, run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduce_training.py --source /path/to/Dimension\ Reduction\ and\ Matrix\ Completion.zip
```

The command checks the frozen source and split checksums, exports temporary disjoint partitions, and fits **only** the previously selected `user_item_bias` model (regularization 5). It verifies the saved training RMSE (0.8166725203779724) with a tolerance of `1e-10`. It **does not** score the test labels or recompute the reported test RMSE. The export coordinator necessarily reads the source archive to partition it; the fit subprocess only receives `fit/` and unlabeled query coordinates. This is data minimization, not verified OS-level isolation. This command was executed successfully in the audit environment on 2026-10-23; cross-platform reproducibility remains to be established.
