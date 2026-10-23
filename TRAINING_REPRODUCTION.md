# Training-only reproduction record

Executed on 2026-10-23 with the original assignment archive, using one BLAS thread:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduce_training.py --source '/mnt/data/Dimension Reduction and Matrix Completion.zip'
```

- Frozen winner: `user_item_bias`, regularization `5.0`.
- Training observations: `70585`.
- Matrix shape: `610 × 9724`.
- Historical training RMSE: `0.8166725203779724`.
- Reproduced training RMSE: `0.8166725203779724`.
- Recorded absolute difference: `0.0`.
- No held-out test metrics were calculated during this reproduction.

This is a **training reproduction in the current environment**, not a clean-install verification or an independent test replication. The exporter reads the full original archive to split it; the model-fitting subprocess does not receive reserved labels. OS-level isolation has not been experimentally verified.
