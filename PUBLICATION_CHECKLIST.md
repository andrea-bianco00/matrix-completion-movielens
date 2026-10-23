# Publication checklist

- [x] Keep source MovieLens archive and held-out labels out of repository.
- [x] Preserve frozen split, selection lock and final results.
- [x] State test result for global winner only.
- [x] State the educational/noncanonical ADMiRA qualification.
- [x] Andrea reports Matteo has consented to GitHub publication (not independently verified).
- [x] Andrea confirms both authors agree to the MIT software license; LICENSE added (coauthor confirmation reported, not independently verified).
- [ ] Make scripts fully path-independent and provide clean dataset acquisition instructions.
- [ ] Verify Docker isolation on a real container runtime, or retain the explicit limitation.
- [x] Remove external training-export dependency from isolated-runner tests; 85/85 algorithm and 55/55 prospective tests pass locally.
- [ ] Make historical experiment scripts fully path-independent; some still rely on audit-era source paths.
- [x] Verify both test suites from the publication directory (140/140, current runtime).
- [ ] Verify a fresh environment with dependencies installed from scratch.
- [ ] Publish on GitHub and then update the portfolio.
- [x] Add offline consistency verifier and tamper-detection tests for selection and final reported metrics (143/143 total tests).
- [ ] Independently reproduce model predictions from original dataset in a clean environment; offline artifact verification alone does not establish this.
- [x] Add a source-path-independent command to reproduce the selected model's training fit and compare its RMSE with the recorded result, without test scoring (local execution passed).

- [x] Add and execute a one-command offline verification entry point (`python check_package.py`), with structured results.
