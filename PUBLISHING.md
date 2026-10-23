# Publishing and authorship

**Authors:** Andrea Bianco and Matteo Mancini (ENSAI university project).

## Publication status

Andrea confirms that Matteo agrees both to publishing the jointly authored project on GitHub and to using the **MIT License** for the code (2026-10-23). This is the author's report, not independent confirmation from Matteo. The repository now includes a `LICENSE` file naming both authors. MIT covers the code in this package, **not** third-party MovieLens data or third-party code with separate terms.

The repository has **not** been published to GitHub by this preparation step. Both authors should review the exact release before making it public.

## Before making a repository public

1. Let both authors review this exact release candidate, README and reported results.
2. Confirm both authors have reviewed the MIT license and final repository contents.
3. Verify no original MovieLens ratings archive, rating CSV, reserved test labels, credentials, or local absolute paths are committed.
4. Run `python check_package.py` in the repository root and save the result.
5. Review the dataset terms on the official GroupLens site; the dataset is not included here.
6. Publish, then link the repository from the portfolio. Avoid claiming that the Docker sandbox or clean-room installation has been verified.

## Suggested GitHub description

Educational comparison of low-rank matrix-completion methods (SVP, proximal SVT and an ADMiRA-inspired variant) on MovieLens, with frozen validation-based selection, transparent baselines and reproducibility checks. Coauthored by Andrea Bianco and Matteo Mancini.

## Suggested portfolio summary

Compared three low-rank matrix-completion approaches for movie ratings with simple predictive baselines. SVP performed best among the tested matrix-completion configurations on validation, while a regularized user–item bias baseline was the overall winner (test RMSE 0.8781). Built reproducible splits, model-selection records, automated tests and integrity checks. Results are specific to the tested configurations; no test scores are claimed for the other methods.
