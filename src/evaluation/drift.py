import json
import numpy as np
from enum import Enum
from pathlib import Path


class DriftLevel(Enum):
    NONE        = "none"
    MILD        = "mild"
    SIGNIFICANT = "significant"
    SEVERE      = "severe"


class DriftDetector:
    N_BINS = 10

    @staticmethod
    def save_reference(df, reference_path):
        """Compute per-feature histograms from reference data and save to JSON."""
        reference = {}
        for col in df.select_dtypes(include='number').columns:
            vals = df[col].dropna().values
            counts, bin_edges = np.histogram(vals, bins=DriftDetector.N_BINS)
            total = counts.sum()
            reference[col] = {
                'bin_edges': bin_edges.tolist(),
                'proportions': (counts / total).tolist() if total > 0 else counts.tolist(),
            }
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
        with open(reference_path, 'w') as f:
            json.dump(reference, f, indent=2)

    @staticmethod
    def compute_psi(reference_path, new_df):
        """Return dict of {feature: PSI} for features present in both reference and new_df."""
        with open(reference_path) as f:
            reference = json.load(f)
        psi_scores = {}
        for col, ref in reference.items():
            if col not in new_df.columns:
                continue
            bin_edges = np.array(ref['bin_edges'])
            expected = np.clip(np.array(ref['proportions']), 1e-8, None)
            vals = new_df[col].dropna().values
            counts, _ = np.histogram(vals, bins=bin_edges)
            total = counts.sum()
            actual = np.clip(counts / total if total > 0 else counts, 1e-8, None)
            psi_scores[col] = float(np.sum((actual - expected) * np.log(actual / expected)))
        return psi_scores

    @staticmethod
    def assess(psi_scores):
        """Return (DriftLevel, mean_psi) based on mean PSI across all features."""
        if not psi_scores:
            return DriftLevel.NONE, 0.0
        mean_psi = float(np.mean(list(psi_scores.values())))
        if mean_psi < 0.1:
            return DriftLevel.NONE, mean_psi
        elif mean_psi < 0.2:
            return DriftLevel.MILD, mean_psi
        elif mean_psi < 0.5:
            return DriftLevel.SIGNIFICANT, mean_psi
        return DriftLevel.SEVERE, mean_psi
