import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from torchvision import transforms

from src.data.augmentation import run_augmentation
from src.evaluation.drift import DriftDetector, DriftLevel
from src.data.loader import ImageDatasetLoader
from src.evaluation.metrics import false_alarm_rate, false_negative_rate, true_negative_rate
from src.evaluation.visualisation import plot_and_log
from src.features.igtd import min_max_transform, select_features_by_variation, table_to_image
from src.models.classifier import CNNTransferLearning
from src.models.dcgan import DCGAN


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _namespace(d):
    """Recursively convert a dict to a SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in d.items()})
    return d


def load_config(path='config/default.yaml'):
    with open(path) as f:
        return _namespace(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_augmentation_stage(cfg, output_path, syn_ratio):
    print(f"Stage 1: Running CTGAN augmentation (syn_ratio={syn_ratio})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_augmentation(
        source_path=cfg.data.source,
        original_samples=cfg.data.original_samples,
        ctgan_epochs=cfg.augmentation.ctgan_epochs,
        ctgan_samples=cfg.augmentation.ctgan_samples,
        syn_ratio=syn_ratio,
        output_path=str(output_path),
    )
    print(f"  Saved augmented data to {output_path}")


def run_igtd_stage(cfg, augmented_path, image_dir):
    print("Stage 2: Running IGTD table-to-image conversion...")
    image_dir.mkdir(parents=True, exist_ok=True)

    num = cfg.features.num_row * cfg.features.num_col
    data = pd.read_csv(augmented_path)
    feature_ids = select_features_by_variation(data, variation_measure=cfg.features.variation_measure, num=num)
    data = data.iloc[:, feature_ids]
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

    result_dir = str(image_dir.parent)
    table_to_image(
        norm_data,
        [cfg.features.num_row, cfg.features.num_col],
        cfg.features.fea_dist_method,
        cfg.features.image_dist_method,
        cfg.features.save_image_size,
        cfg.features.max_step,
        cfg.features.val_step,
        result_dir,
        cfg.features.error,
    )
    print(f"  Images saved to {image_dir}")


def _save_drift_reference(cfg, augmented_path):
    num = cfg.features.num_row * cfg.features.num_col
    data = pd.read_csv(augmented_path)
    feature_ids = select_features_by_variation(
        data, variation_measure=cfg.features.variation_measure, num=num
    )
    data = data.iloc[:, feature_ids]
    DriftDetector.save_reference(data, cfg.drift.reference_path)
    print(f"  Drift reference saved to {cfg.drift.reference_path}")


def run_dcgan_stage(cfg, image_dir, weights_path, num_epochs, plots_save_dir):
    print(f"Stage 3: Training DCGAN ({num_epochs} epochs)...")
    transform = transforms.Compose([
        transforms.Resize(cfg.dcgan.img_size),
        transforms.ToTensor(),
    ])
    dataset = ImageDatasetLoader(str(image_dir), image_type='png', transform=transform)
    dcgan = DCGAN(
        dataset,
        nz=cfg.dcgan.latent_dim,
        ngf=cfg.dcgan.ngf,
        ndf=cfg.dcgan.ndf,
        nc=cfg.dcgan.img_channels,
        weights_save_path=str(weights_path),
        plots_save_dir=plots_save_dir,
    )
    dcgan.train(num_epochs=num_epochs)
    print(f"  Discriminator weights saved to {weights_path}")


def _append_training_log(cfg, grid_search, scoring, syn_ratio, dcgan_epochs):
    log_path = cfg.outputs.training_log
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['timestamp', 'syn_ratio', 'dcgan_epochs', 'best_lr'] + list(scoring.keys())
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'syn_ratio': syn_ratio,
        'dcgan_epochs': dcgan_epochs,
        'best_lr': grid_search.best_params_.get('learning_rate', ''),
    }
    for scorer_name in scoring.keys():
        scores = grid_search.cv_results_[f'mean_test_{scorer_name}']
        row[scorer_name] = round(float(scores[grid_search.best_index_]), 4)
    write_header = not Path(log_path).exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Training log updated: {log_path}")


def run_classification_stage(cfg, image_dir, weights_path, augmented_path,
                              plots_dir, tensorboard_dir, syn_ratio, dcgan_epochs):
    print("Stage 4: Training CNN classifier...")
    transform = transforms.Compose([
        transforms.Resize(cfg.dcgan.img_size),
        transforms.ToTensor(),
    ])
    image_dataset = ImageDatasetLoader(str(image_dir), image_type='png', transform=transform)

    # Rebuild label vector aligned with sorted image filenames
    num = cfg.features.num_row * cfg.features.num_col
    data = pd.read_csv(augmented_path)
    feature_ids = select_features_by_variation(data, variation_measure=cfg.features.variation_measure, num=num)
    data = data.iloc[:, feature_ids]
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)
    labels = torch.tensor(norm_data['Label'].values).long()

    # Load pretrained discriminator
    discriminator = DCGAN.Discriminator(cfg.dcgan.img_channels, cfg.dcgan.ndf)
    discriminator.load_state_dict(torch.load(str(weights_path), map_location='cpu'))
    discriminator.eval()

    num_classes = int(labels.max().item()) + 1
    cnn_model = CNNTransferLearning(
        discriminator,
        num_classes=num_classes,
        img_channels=cfg.dcgan.img_channels,
        img_size=cfg.dcgan.img_size,
        learning_rate=cfg.classifier.learning_rates[0],
    )
    if cfg.classifier.unfreeze_blocks > 0:
        cnn_model.set_trainable_layers(cfg.classifier.unfreeze_blocks)
        print(f"  Unfreezing {cfg.classifier.unfreeze_blocks} discriminator block(s)")

    scoring = {
        'precision':          make_scorer(precision_score, average='macro', zero_division=1),
        'recall':             make_scorer(recall_score, average='macro'),
        'f1_score':           make_scorer(f1_score, average='macro'),
        'accuracy':           make_scorer(accuracy_score),
        'false_alarm_rate':   make_scorer(false_alarm_rate),
        'false_negative_rate': make_scorer(false_negative_rate),
        'true_negative_rate': make_scorer(true_negative_rate),
    }
    param_grid = {'learning_rate': cfg.classifier.learning_rates}
    grid_search = GridSearchCV(
        cnn_model, param_grid,
        cv=cfg.classifier.cv_folds,
        scoring=scoring,
        refit='f1_score',
        error_score='raise',
    )
    grid_search.fit(image_dataset, labels)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best F1:     {grid_search.best_score_:.4f}")

    plot_and_log(
        grid_search, scoring, image_dataset, labels,
        cv=cfg.classifier.cv_folds,
        plots_dir=plots_dir,
        tensorboard_dir=tensorboard_dir,
    )
    _save_drift_reference(cfg, augmented_path)
    _append_training_log(cfg, grid_search, scoring, syn_ratio, dcgan_epochs)


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------

def check_drift(cfg, new_data_path):
    reference_path = cfg.drift.reference_path
    if not Path(reference_path).exists():
        print("No drift reference found. Run the full pipeline first to establish a baseline.")
        return

    data = pd.read_csv(new_data_path)
    num = cfg.features.num_row * cfg.features.num_col
    feature_ids = select_features_by_variation(
        data, variation_measure=cfg.features.variation_measure, num=num
    )
    data = data.iloc[:, feature_ids]

    psi_scores = DriftDetector.compute_psi(reference_path, data)
    level, mean_psi = DriftDetector.assess(psi_scores)

    print(f"\nDrift Report — {new_data_path}")
    print(f"  Mean PSI:    {mean_psi:.4f}")
    print(f"  Drift Level: {level.value.upper()}")

    top5 = sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top drifted features:")
    for feat, psi in top5:
        print(f"    {feat}: {psi:.4f}")

    recommendations = {
        DriftLevel.NONE:        "No retraining needed.",
        DriftLevel.MILD:        "Retrain head only: Stage 4 always runs, no config change needed.",
        DriftLevel.SIGNIFICANT: "Set classifier.unfreeze_blocks=1 in config, delete weights file, re-run.",
        DriftLevel.SEVERE:      "Full retrain: delete outputs/images/data/ and weights/, then re-run.",
    }
    print(f"\n  Recommendation: {recommendations[level]}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main(config_path='config/default.yaml'):
    cfg = load_config(config_path)
    weights_dir = Path(cfg.outputs.weights_dir)
    n_combos = len(cfg.sweep.syn_ratios) * len(cfg.sweep.dcgan_epochs)
    combo = 0

    for syn_ratio in cfg.sweep.syn_ratios:
        augmented_path = Path(cfg.outputs.augmented) / f"{syn_ratio}_sampled.csv"
        image_dir = Path(cfg.outputs.images) / str(syn_ratio) / 'data'

        # Stage 1: per syn_ratio — skip if CSV already exists
        if not augmented_path.exists():
            run_augmentation_stage(cfg, augmented_path, syn_ratio)
        else:
            print(f"Stage 1 [{syn_ratio}]: Skipped (found {augmented_path})")

        # Stage 2: per syn_ratio — skip if images already exist
        if not image_dir.exists() or not any(image_dir.glob('*.png')):
            run_igtd_stage(cfg, augmented_path, image_dir)
        else:
            print(f"Stage 2 [{syn_ratio}]: Skipped (found images in {image_dir})")

        for dcgan_epochs in cfg.sweep.dcgan_epochs:
            combo += 1
            print(f"\n[{combo}/{n_combos}] syn_ratio={syn_ratio}, dcgan_epochs={dcgan_epochs}")
            weights_path = weights_dir / f"dcgan_{syn_ratio}_{dcgan_epochs}.pth"
            plots_dir = f"{cfg.outputs.plots}/{syn_ratio}/{dcgan_epochs}"
            tensorboard_dir = f"{cfg.outputs.tensorboard}/{syn_ratio}/{dcgan_epochs}"

            # Stage 3: per (syn_ratio, dcgan_epochs) — skip if weights already exist
            if not weights_path.exists():
                run_dcgan_stage(cfg, image_dir, weights_path, dcgan_epochs, plots_dir)
            else:
                print(f"Stage 3: Skipped (found {weights_path})")

            # Stage 4: always run
            run_classification_stage(
                cfg, image_dir, weights_path, augmented_path,
                plots_dir, tensorboard_dir, syn_ratio, dcgan_epochs
            )


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--check-drift':
        check_drift(load_config(), sys.argv[2])
    else:
        config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/default.yaml'
        main(config_path)
