# VenusPheno

VenusPheno predicts bacterial phenotypes from genome-scale collections of
precomputed protein embeddings.  It uses a masked Set Transformer to aggregate
protein-level ProtT5 embeddings into genome representations, followed by either
single-task learning (STL) or masked, task-balanced multi-task learning (MTL).

The repository also contains the leave-one-task-out (LOTO) transfer analysis
used to construct directed task-transfer graphs and select task groups for
transfer-guided MTL.

## Repository scope

This release contains the code used for the paper's model training,
evaluation, and task-grouping analyses.  It intentionally does **not** contain
raw genomes, phenotype labels, protein embeddings, trained checkpoints, or
large experiment outputs.  Before submission, add a persistent public URL for
the versioned metadata and split files in the **Data availability** section
below; exact reproduction requires those files as well as this code.

## Installation

Use Python 3.10+ and a PyTorch build appropriate for the available CUDA
runtime.  From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Run the lightweight padding-mask check before training:

```bash
python -m unittest discover -s tests -v
python src/training/test_padding_mask.py
```

## Input data format

Each genome is represented by one HDF5 file with these datasets:

| Dataset | Shape | Description |
| --- | --- | --- |
| `Embeddings` | `(n_proteins, 1024)` | Float32 ProtT5 embedding for each protein. |
| `Names` | `(n_proteins,)` | UTF-8 protein identifiers. |

The STL metadata CSV requires `gcf`, `h5_path`, and binary `label` columns.
The grouped-MTL task table is a TSV with the columns shown in
[`configs/panel_v2_example.tsv`](configs/panel_v2_example.tsv); each per-task
metadata CSV uses the STL schema.  Every task's train, validation, and test ID
files contain one genome identifier per line and must be mutually disjoint.

## Reproducing the training protocol

The primary analyses use a frozen 14-task panel, a genus-disjoint
train/validation/test partition, masked binary cross-entropy, per-task loss
means, and validation-only checkpoint selection.  Test labels are not used for
early stopping or task-group selection.

First validate a completed task table:

```bash
PYTHONPATH=src python src/training/train_dynamic_group_panel_v2.py \
  --strategy baseline --batch_size 32 --seed 42 \
  --tasks motility,sporulation,gram_stain,cell_shape__bacillus,cell_shape__coccus,metabolism__aerobic,metabolism__anaerobic,metabolism__facultative,carb_citrate,carb_glucose,carb_glycerol,carb_lactose,carb_maltose,carb_sucrose \
  --task_table configs/panel_v2_example.tsv --save_root results/preflight \
  --check_splits
```

Train the matched hard-sharing baseline:

```bash
PYTHONPATH=src python src/training/train_dynamic_group_panel_v2.py \
  --strategy baseline --model_type hps_equivalent --batch_size 32 --seed 42 \
  --epochs 80 --lr 1e-4 --weight_decay 1e-4 --grad_clip_norm 1.0 \
  --seed_orth_lambda 0.01 --task_table configs/panel_v2_example.tsv \
  --save_root results/hardshare_seed42
```

For STL, supply one task's metadata and fixed split files:

```bash
PYTHONPATH=src python src/training/train_stl_panel.py \
  --metadata data/metadata/motility.csv \
  --train_txt data/splits/motility_train.txt \
  --val_txt data/splits/motility_val.txt \
  --test_txt data/splits/motility_test.txt \
  --mode isab_pma --num_heads 4 --num_inds 64 --num_seeds 4 \
  --hidden_dim 512 --dropout 0.3 --head_type res_mlp \
  --optimizer adamw --weight_decay 1e-4 --grad_clip_norm 1.0 \
  --checkpoint_metric bacc --decision_threshold 0.5 --seed 42 \
  --save_root results/stl_motility_seed42
```

For transfer-guided MTL, run the full-panel model and matched LOTO models
(each omitting one source task), export validation metrics with
`evaluate_grouped_mtl_validation_thresholds.py`, construct the transfer graph
with the scripts in `analysis/`, and train the resulting fixed task groups.
The analysis scripts retain the validation-only graph-selection protocol used
in the paper.

## Key files

```text
src/models/                 Set Transformer aggregators and prediction heads
src/training/               STL, grouped MTL, evaluation, and summary scripts
src/utils/                  HDF5 datasets, metrics, logging, and early stopping
analysis/                   LOTO transfer-graph and partition analyses
configs/                    Example panel schema and paper task-group assignments
tests/                      Lightweight model smoke test
```

## Data availability

**Before submission, replace this sentence with the final public data URL and
version/DOI.**  The repository must point to public, versioned phenotype
metadata and the exact genus-disjoint split files used in the paper.  ProtT5
embeddings may be regenerated from the referenced protein sequences, or a
versioned embedding archive may be supplied separately.

## License

Released under the [MIT License](LICENSE).
