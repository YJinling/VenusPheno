# Project Structure Notes

## Source code

- `src/venuspheno/models/aggregator.py`: set-transformer style aggregation modules and bag pooling logic.
- `src/venuspheno/models/classifier.py`: single-task, multitask, and no-aggregation classifiers.
- `src/venuspheno/utils/dataset.py`: dataset loaders and collate functions for bag-level and embedding-level inputs.
- `src/venuspheno/utils/metrics.py`: binary and multitask evaluation metrics.
- `src/venuspheno/utils/logger.py`: logging helpers.
- `src/venuspheno/utils/early_stopping.py`: early stopping helper.

## Training scripts

- `scripts/binary/train_0225.py`: binary bag-level training for standard aggregation experiments.
- `scripts/binary/train_0310_attn.py`: binary bag-level training with optional attention extraction.
- `scripts/multitask/train_pheno_multitask.py`: bag-level multitask phenotype training for a smaller task subset.
- `scripts/multitask/train_pheno_multitask2.py`: bag-level multitask phenotype training for a larger carbohydrate task set.
- `scripts/baselines/train_noagg.py`: genome-level embedding baseline without bag aggregation.
- `scripts/baselines/train_noagg_multitask.py`: multitask embedding baseline without bag aggregation.

## Suggested GitHub usage

- Keep code under `src/` and `scripts/`.
- Put metadata examples or small samples under `data/processed/`.
- Do not commit large raw data files or training outputs.
