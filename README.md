# dcgan-cnn-ids

A hybrid **DCGAN + CNN pipeline for Network Intrusion Detection (IDS)**, targeting polymorphic exploit detection. Network flow features are converted to images, a DCGAN learns behavioral representations, and its discriminator is used as a frozen feature extractor for a transfer-learning classifier.

## Pipeline

```
CSV dataset
  → CTGAN augmentation       (address class imbalance)
  → IGTD table-to-image      (78 features → 26×3 pixel PNG per flow)
  → DCGAN training           (unsupervised representation learning)
  → CNN transfer learning    (frozen discriminator + trainable head)
  → GridSearchCV + metrics   (precision, recall, F1, FAR, FNR, TNR)
```

## Setup

```bash
# Recommended
conda env create -f environment.yml

# Alternative
pip install -r requirements.txt
```

**Required data** (not included): place the dataset at `./data/clean/cleaned_ids2018_sampled.csv`

## Usage

```bash
# Run pipeline (single config)
python main.py

# Explicit config path
python main.py config/default.yaml

# Check for data drift against the training baseline
python main.py --check-drift path/to/new_data.csv
```

All hyperparameters are in `config/default.yaml` — edit that file, not the code.

## Sweep / Ablation

To run multiple configurations in one pass, extend the sweep lists in `config/default.yaml`:

```yaml
sweep:
  syn_ratios: [0.8, 0.9, 1.0]
  dcgan_epochs: [10, 50, 100]
```

This runs all 9 combinations. Each has isolated outputs (images, weights, plots). Results are appended to `outputs/training_log.csv`.

Stages 1–3 are skipped automatically if their outputs already exist:

| Stage | Skip condition | Output to delete to re-run |
|-------|---------------|---------------------------|
| 1 CTGAN | CSV exists | `outputs/augmented/{ratio}_sampled.csv` |
| 2 IGTD | Images exist | `outputs/images/{ratio}/data/` |
| 3 DCGAN | Weights exist | `weights/dcgan_{ratio}_{epochs}.pth` |
| 4 CNN | Never skipped | — |

## Drift Detection

After each full run, a reference distribution is saved to `outputs/drift_reference.json`. To check whether new traffic data has drifted from the training distribution:

```bash
python main.py --check-drift new_traffic.csv
```

Output includes mean PSI, drift level (NONE / MILD / SIGNIFICANT / SEVERE), the top 5 most-drifted features, and a recommended retraining action.

### Tiered Retraining

| Drift level | PSI | Action |
|-------------|-----|--------|
| NONE | < 0.10 | No retraining needed |
| MILD | 0.10–0.20 | Re-run Stage 4 (always runs) |
| SIGNIFICANT | 0.20–0.50 | Set `classifier.unfreeze_blocks: 1`, delete weights, re-run |
| SEVERE | ≥ 0.50 | Full retrain — delete images and weights |

`classifier.unfreeze_blocks` controls how many discriminator conv blocks are unfrozen during fine-tuning (0 = fully frozen, 1 = last block only, up to 4 = all blocks).

## Project Structure

```
config/default.yaml          # All hyperparameters
src/
  data/
    preprocessing.py         # Feature type detection, null removal, dataset combining
    loader.py                # ImageDatasetLoader (sorted, PyTorch-compatible)
    augmentation.py          # CTGAN synthetic data generation
  features/
    igtd.py                  # Table-to-image conversion (IGTD algorithm)
  models/
    dcgan.py                 # DCGAN (Generator + Discriminator)
    classifier.py            # CNNTransferLearning (sklearn-compatible, tiered unfreezing)
  evaluation/
    metrics.py               # FAR, FNR, TNR
    visualisation.py         # Score plots, learning curves, TensorBoard
    drift.py                 # PSI-based drift detection (DriftDetector, DriftLevel)
weights/                     # Trained discriminator weights (per sweep combination)
outputs/
  augmented/                 # CTGAN-augmented CSVs
  images/                    # IGTD-generated PNGs (per syn_ratio)
  plots/                     # Score and learning curve plots (per combination)
  tensorboard/               # TensorBoard logs
  training_log.csv           # Per-combination metrics across all sweep runs
  drift_reference.json       # PSI reference distribution
archive/                     # Superseded code kept for reference
```
