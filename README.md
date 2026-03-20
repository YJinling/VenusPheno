# VenusPheno

VenusPheno is a cleaned-up research repository for phenotype prediction experiments on genome or protein embedding data.

The code in this repository was originally developed as experiment scripts. It has been reorganized into a clearer GitHub-friendly layout without changing the core training logic.

## Repository structure

```text
VenusPheno/
├── data/                 # data placeholders and dataset notes
├── docs/                 # project notes
├── outputs/              # default training outputs
├── scripts/              # runnable training scripts
│   ├── baselines/
│   ├── binary/
│   └── multitask/
└── src/venuspheno/       # reusable code package
    ├── models/
    └── utils/
```

## Script overview

- `scripts/binary/`: binary classification training, including attention extraction experiments.
- `scripts/multitask/`: bag-level multitask phenotype training.
- `scripts/baselines/`: baseline models using genome-level embeddings without bag aggregation.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Run scripts from the repository root, for example:

```bash
python scripts/binary/train_0225.py --help
python scripts/multitask/train_pheno_multitask2.py --help
```

## Notes

- Several scripts still contain local absolute paths in their default dataset arguments because they came from active experiments.
- Before sharing publicly, replace those paths with your own dataset locations or convert them into config files.
- Training outputs now default to `outputs/`.
