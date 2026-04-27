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
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "MedHouseVal"

BASE_DIR = Path(__file__).resolve().parent
MODEL_1_DIR = BASE_DIR / "Models" / "Model 1"
TRAIN_DIR = BASE_DIR / "Dataset" / "Train"
TEST_DIR = BASE_DIR / "Dataset" / "Test"
SKLEARN_DATA_DIR = BASE_DIR / "Dataset" / "sklearn_data"
RESULTS_DIR = BASE_DIR / "Results"


def create_directories():
    """Create output directories."""
    for directory in [MODEL_1_DIR, TRAIN_DIR, TEST_DIR, SKLEARN_DATA_DIR,
                      RESULTS_DIR]:
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


def save_train_test_data(df, feature_names):
    """Split and save train/test datasets."""
    x = df[feature_names]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_df = x_train.copy()
    train_df[TARGET_COLUMN] = y_train
    test_df = x_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(TRAIN_DIR / "train.csv", index=False)
    test_df.to_csv(TEST_DIR / "test.csv", index=False)

    return x_train, x_test, y_train, y_test


def grid_results_to_table(grid_search):
    """Convert GridSearchCV results into a readable CV-RMSE table."""
    results = pd.DataFrame(grid_search.cv_results_)
    param_columns = [col for col in results.columns if col.startswith("param_")]
    table = results[param_columns + ["mean_test_score", "std_test_score"]].copy()
    table["CV-RMSE"] = -table["mean_test_score"]
    table["CV-RMSE std"] = table["std_test_score"]
    table = table.drop(columns=["mean_test_score", "std_test_score"])
    table = table.sort_values("CV-RMSE").reset_index(drop=True)
    table.columns = [col.replace("param_", "") for col in table.columns]
    return table


def train_ridge_model(x_train, y_train):
    """Train Ridge Regression with scaling and 4-fold CV."""
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
    ridge_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}

    ridge_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=ridge_grid,
        cv=4,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    ridge_search.fit(x_train, y_train)

    joblib.dump(ridge_search.best_estimator_,
                MODEL_1_DIR / "ridge_pipeline.joblib")

    ridge_results = grid_results_to_table(ridge_search)
    ridge_results.to_csv(MODEL_1_DIR / "ridge_grid_results.csv", index=False)
    return ridge_search


def main():
    """Run feature analysis, split data, and train Ridge Regression."""
    create_directories()
    df, feature_names = load_dataset()
    analyze_features(df, feature_names)
    x_train, _, y_train, _ = save_train_test_data(df, feature_names)
    train_ridge_model(x_train, y_train)
    print("Ridge baseline trained and saved.")


if __name__ == "__main__":
    main()
