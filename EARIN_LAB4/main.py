"""Lab 4: California Housing feature analysis."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

try:
    import certifi
except ImportError:
    certifi = None

if certifi is not None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing


TARGET_COLUMN = "MedHouseVal"

BASE_DIR = Path(__file__).resolve().parent
SKLEARN_DATA_DIR = BASE_DIR / "Dataset" / "sklearn_data"
RESULTS_DIR = BASE_DIR / "Results"


def create_directories():
    """Create output directories."""
    for directory in [SKLEARN_DATA_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load California Housing dataset as a pandas DataFrame."""
    housing = fetch_california_housing(
        data_home=SKLEARN_DATA_DIR,
        as_frame=True,
    )
    return housing.frame.copy(), housing.feature_names


def analyze_features(df, feature_names):
    """Save summary statistics, feature correlations, and heatmap."""
    summary = df[feature_names + [TARGET_COLUMN]].describe()
    summary.to_csv(RESULTS_DIR / "feature_summary_statistics.csv")

    correlation_matrix = df[feature_names].corr()
    correlation_matrix.to_csv(RESULTS_DIR / "feature_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    ax.set_title("Feature Correlation Matrix")

    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            value = correlation_matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    retention_decision = (
        "No fully dependent or obviously redundant features were found, so all "
        "eight features were kept."
    )
    (RESULTS_DIR / "feature_retention_decision.txt").write_text(
        retention_decision + "\n",
        encoding="utf-8",
    )


def main():
    """Run the feature-analysis workflow."""
    create_directories()
    df, feature_names = load_dataset()
    analyze_features(df, feature_names)
    print("Feature analysis files saved to the Results folder.")


if __name__ == "__main__":
    main()
