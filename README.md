# Matrix Completion on MovieLens: algorithms, choices and evidence

**Andrea Bianco · Matteo Mancini** — ENSAI university project, revised as a reproducible educational benchmark.

> **Main finding:** Among the three implemented matrix-completion methods, **SVP** had the lowest validation RMSE. However, a much simpler **regularized user–item bias baseline** performed better than all three. The baseline was selected **before opening the final test**, where it obtained **RMSE 0.878057** on 15,126 held-out ratings. This is a result for our implementations and experimental setup, **not** a universal ranking of algorithms.

## The problem and why it matters

Movie ratings form a matrix (R\) whose rows are users and columns are films. Only a small fraction of its entries are observed. **Matrix completion** asks whether we can predict the missing entries by exploiting structure in the observed ratings.

A common hypothesis is that user preferences and movie characteristics can be summarized by a small number of latent dimensions, making the rating matrix approximately **low-rank**. The challenge is to learn that structure without simply memorizing the observed ratings.

**Research question:** How do three low-rank reconstruction approaches compare under a common split and fixed computational budgets? And do they beat simple, interpretable baselines?

## Why these models?

| Family | Core idea | Why include it? | Important qualification |
| --- | --- | --- | --- |
| **SVP** | Gradient steps on observed-rating error, followed by a projection to a fixed matrix rank | Directly tests the low-rank hypothesis with an explicit rank constraint | We compare ranks 2, 5 and 10; convergence was not certified |
| **SVT** | Gradient/proximal updates with singular-value shrinkage | Studies a softer, nuclear-norm-based alternative to a hard rank constraint | Our implementation is **nuclear-norm proximal gradient**, not necessarily the original dual SVT algorithm |
| **ADMiRA-inspired variant** | Iterative identification of low-rank components | Explores a pursuit-style alternative | This is an **educational lab variant**, **not** a verified implementation of canonical ADMiRA |
| **Global mean** | Predict the training-set average for every rating | A minimal reference: any useful model should beat it | Cannot model users or films separately |
| **User–item bias** | Predict training mean + user effect + film effect, with regularization | Tests whether simple systematic rating differences explain much of the signal | Does not learn user–film interactions |

The bias baseline predicts

`prediction(user, movie) = training_mean + user_bias + movie_bias`.

The biases are fitted **only on training ratings**, using squared-error loss plus a penalty on bias magnitude. Unseen users or films receive a zero bias. We tested regularization values **5** and **20**, rather than choosing the best value on the test set.

## Why this experimental design?

**Data.** MovieLens Latest Small contains **100,836 ratings**, from **610 users** and **9,724 rated films**. The data are **not redistributed** in this repository; obtain the original archive from [GroupLens](https://grouplens.org/datasets/movielens/latest/).

**Fixed rating-wise split.** We use **70,585 training**, **15,125 validation**, and **15,126 test** ratings (approximately 70/15/15). Training fits model parameters; validation compares the predefined candidates; the held-out test evaluates **only the globally selected winner**. The split indices and checksums are frozen in [the split manifest](prospective/split_manifest_18.json).

**Why rating-wise?** It measures prediction for randomly withheld ratings under a known user/film catalog. It does **not** measure performance on future ratings or wholly new users. Some films appear in the catalog but have no training ratings, so we report **warm-item** and **cold-item** performance separately.

**Why fixed grids and budgets?** A predefined [benchmark plan](prospective/benchmark_plan_28.json) reduces the temptation to adjust settings after seeing validation or test scores. We compared **11 candidates**: one global mean, two bias settings, three SVP ranks, three SVT thresholds and two ADMiRA-variant ranks. The solver budgets were fixed (SVP 100 iterations, SVT 50, ADMiRA variant 100); a 120-second per-fit cap and one BLAS thread were specified. These budgets make the comparison reproducible **within this protocol**, but do not guarantee equally optimized implementations.

**Why raw RMSE?** Root mean squared error is an interpretable prediction metric in rating units and penalizes larger errors more strongly. We selected models by **unclipped validation RMSE**. Clipping predictions to the rating range was only reported descriptively after selection.

## What did we observe?

**Best configuration in each family, selected on validation:**

| Model | Chosen setting | Validation RMSE |
| --- | --- | ---: |
| **User–item bias baseline** | Regularization 5 | **0.860652** |
| Global mean | — | 1.035442 |
| **SVP** | Rank 5 | **1.389501** |
| ADMiRA-inspired variant | Rank 2 | 1.490713 |
| SVT proximal implementation | Threshold 30 | 1.719120 |

Source: [frozen validation results](prospective/validation_results_34.json) and [pre-test selection lock](prospective/selection_lock_41.json).

**Interpretation, without overstating it:**

1. **SVP wins among the three low-rank methods tested.** A direct rank constraint worked better here than the tested shrinkage and pursuit-style alternatives. This does **not** establish that SVP is inherently superior.
2. **The bias baseline wins overall.** User-specific rating tendencies and film-specific rating levels capture substantial predictive signal in this split. A more sophisticated model is not automatically a better predictor.
3. **Training fit is not generalization.** For SVP, increasing rank can improve training error while worsening validation error; this is consistent with overfitting. It is not proof that overfitting is the sole reason for the results.
4. **Optimization matters.** The selected SVP, SVT and ADMiRA-variant fits reached their iteration limits; we do **not** claim convergence to optimal solutions. Different implementations, tuning budgets or splits could change the ranking.

**Final test — only the preselected overall winner:**

| Metric | User–item bias, regularization 5 |
| --- | ---: |
| Test RMSE, **unclipped** | **0.878057** |
| Warm-item RMSE (14,425 ratings) | 0.871318 |
| Cold-item RMSE (701 ratings) | 1.006776 |
| Clipped test RMSE (descriptive only) | 0.878004 |

Source: [one-shot final winner result](prospective/final_winner_results_43.json). **We did not evaluate the other model families on the final test.** The test has already been opened and must not be used for further tuning.

## What the results do *not* prove

- This is **one MovieLens dataset** and a **rating-wise** split, not a chronological or cold-user deployment evaluation.
- The candidate grids are **limited** and the algorithms did **not** all reach certified convergence. The outcome is a comparison of these configurations and budgets, not a definitive algorithm leaderboard.
- The educational ADMiRA variant must **not** be presented as a canonical ADMiRA reproduction.
- Earlier exploratory audits also examined MovieLens under other splits, so the dataset is **not entirely independent** of the development process.
- The held-out test result is available only for the preselected baseline. Reporting other models' validation scores alongside its test score must not be mistaken for a test-set comparison.
- The fit worker receives training data and unlabeled query coordinates; the Docker launcher was **not validated in a real container**. OS-level isolation is **not certified**.

## Reproduce and inspect the evidence

**Requirements:** Python, NumPy and SciPy; see [requirements.txt](requirements.txt).

```bash
python -m pip install -r requirements.txt
python check_package.py
```

The check runs the local algorithm/protocol tests and verifies recorded release evidence **without downloading MovieLens or reopening the test**. Passing does not prove cross-platform reproduction or re-run the historical experiment.

To check the frozen results independently:

```bash
python verify_release.py
python -m unittest test_verify_release -q
```

To reproduce **the selected baseline's training fit**, supply the unchanged original assignment ZIP containing MovieLens `ratings.csv`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduce_training.py --source "/path/to/Dimension Reduction and Matrix Completion.zip"
```

This verifies the frozen source/split hashes and the recorded training RMSE **0.8166725204**. It **does not** rescore the held-out test. Some historical experiment scripts retain audit-era file paths and are included for traceability, not as a turnkey full-grid runner.

### Evidence map

| Question | Where to look |
| --- | --- |
| Which models and parameters were planned? | [Benchmark plan](prospective/benchmark_plan_28.json) |
| How was the data partitioned? | [Split manifest](prospective/split_manifest_18.json) |
| What happened on validation? | [Validation results](prospective/validation_results_34.json) |
| Was the winner frozen before test? | [Selection lock](prospective/selection_lock_41.json) |
| What was actually measured on test? | [Final winner result](prospective/final_winner_results_43.json) |
| Can the saved evidence be checked? | [Release verifier](verify_release.py), [verification notes](RELEASE_VERIFICATION.md) |
| How can the training fit be reproduced? | [Training reproduction](TRAINING_REPRODUCTION.md) |
| What remains unverified? | [Publication checklist](PUBLICATION_CHECKLIST.md) |

## Authorship, license and data

**Authors:** Andrea Bianco and Matteo Mancini. This repository is a revised, tested educational/research implementation stemming from an ENSAI university project. Code is licensed under [MIT](LICENSE), with both authors' agreement. **MovieLens data are not included and are not licensed by our MIT license.** See [publishing notes](PUBLISHING.md).

**Bottom line:** In this carefully bounded experiment, **SVP performed best among the three low-rank approaches**, while a **regularized user–item bias baseline was the strongest overall predictor**. The meaningful lesson is methodological: establish simple baselines, separate model selection from final evaluation, and state the limits of the evidence.
