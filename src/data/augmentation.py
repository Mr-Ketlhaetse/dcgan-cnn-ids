from pathlib import Path

import pandas as pd
from ctgan import CTGAN
from src.data.preprocessing import detect_features, remove_null_rows, combine_datasets


def run_augmentation(source_path, original_samples, ctgan_epochs, ctgan_samples,
                     syn_ratio, output_path):
    """
    Load raw data, train CTGAN, generate synthetic samples, and save the combined dataset.

    Args:
        source_path (str): Path to the original CSV dataset.
        original_samples (int): Number of rows to load from the source.
        ctgan_epochs (int): CTGAN training epochs (use >=100 for meaningful synthesis).
        ctgan_samples (int): Number of synthetic rows to generate.
        syn_ratio (float): Fraction of rows drawn from the real pool for the combined set.
        output_path (str): Destination path for the combined CSV.

    Returns:
        pd.DataFrame: The combined dataset.
    """
    real_data = pd.read_csv(source_path).iloc[:original_samples]
    real_data = remove_null_rows(real_data)

    continuous_features, discrete_features = detect_features(real_data)

    ctgan = CTGAN(epochs=ctgan_epochs)
    ctgan.fit(real_data, discrete_features)
    synthetic_data = ctgan.sample(ctgan_samples)

    filename = output_path.split('/')[-1]
    folder = '/'.join(output_path.split('/')[:-1])
    combined, _ = combine_datasets(real_data, synthetic_data, syn_ratio, filename)

    # combine_datasets saves to its own hardcoded folder; ensure file lands at output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return combined
