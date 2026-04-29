"""Lab 4: California Housing price prediction.

Run this file to create datasets, train models, and save result files.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

try:
    import certifi
except ImportError:
    certifi = None

if certifi is not None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import joblib
import matplotlib
from joblib import parallel_backend

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
TARGET_COLUMN = "MedHouseVal"

BASE_DIR = Path(__file__).resolve().parent
MODEL_1_DIR = BASE_DIR / "Models" / "Model 1"
MODEL_2_DIR = BASE_DIR / "Models" / "Model 2"
MODEL_3_DIR = BASE_DIR / "Models" / "Model 3"
TRAIN_DIR = BASE_DIR / "Dataset" / "Train"
TEST_DIR = BASE_DIR / "Dataset" / "Test"
SKLEARN_DATA_DIR = BASE_DIR / "Dataset" / "sklearn_data"
RESULTS_DIR = BASE_DIR / "Results"


def create_directories():
    """Create project output directories."""
    for directory in [
        MODEL_1_DIR,
        MODEL_2_DIR,
        MODEL_3_DIR,
        TRAIN_DIR,
        TEST_DIR,
        SKLEARN_DATA_DIR,
        RESULTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load California Housing dataset as a pandas DataFrame."""
    housing = fetch_california_housing(
        data_home=SKLEARN_DATA_DIR,
        as_frame=True,
    )
    feature_names = housing.feature_names
    df = housing.frame.copy()
    return df, feature_names


def analyze_features(df, feature_names):
    """Save feature summary statistics, correlations, and heatmap."""
    print("Generating stats and heatmap...")

    summary = df[feature_names + [TARGET_COLUMN]].describe()
    summary.to_csv(RESULTS_DIR / "feature_summary_statistics.csv")

    correlation_matrix = df[feature_names].corr()
    correlation_matrix.to_csv(RESULTS_DIR / "feature_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 10},
    )

    ax.set_title("Feature Correlation Matrix (Target Excluded)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    heatmap_path = RESULTS_DIR / "feature_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)

    return summary, correlation_matrix, heatmap_path


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


def tune_and_train_models(x_train, y_train):
    """Tune both models with 4-fold CV using RMSE as the selection metric."""
    print("Training Model 1: Ridge Regression grid search...", flush=True)
    ridge_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    ridge_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}

    ridge_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=ridge_grid,
        cv=4,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_search.fit(x_train, y_train)
    print("Model 1 complete.", flush=True)

    print("Training Model 2: Random Forest grid search...", flush=True)
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    rf_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    rf_search = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_grid,
        cv=4,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    rf_search.fit(x_train, y_train)
    print("Model 2 complete.", flush=True)

    print("Training Model 3: XGBoost grid search...", flush=True)
    xgb_model = XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=1, objective="reg:squarederror"
    )

    xgb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }

    xgb_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_grid,
        cv=4,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    xgb_search.fit(x_train, y_train)
    print("Model 3 complete.", flush=True)

    joblib.dump(ridge_search.best_estimator_, MODEL_1_DIR / "ridge_pipeline.joblib")
    joblib.dump(rf_search.best_estimator_, MODEL_2_DIR / "random_forest_model.joblib")
    joblib.dump(xgb_search.best_estimator_, MODEL_3_DIR / "xgboost_model.joblib")

    ridge_results = grid_results_to_table(ridge_search)
    rf_results = grid_results_to_table(rf_search)
    xgb_results = grid_results_to_table(xgb_search)

    ridge_results.to_csv(MODEL_1_DIR / "ridge_grid_results.csv", index=False)
    rf_results.to_csv(MODEL_2_DIR / "random_forest_grid_results.csv", index=False)
    xgb_results.to_csv(MODEL_3_DIR / "xgboost_grid_results.csv", index=False)

    return ridge_search, rf_search, xgb_search


def evaluate_model(y_test, predictions):
    """Calculate standard regression metrics for a fitted model."""
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }


def create_comparison_table(
    ridge_search, rf_search, xgb_search, ridge_preds, rf_preds, xgb_preds, y_test
):
    """Create and save final model comparison table."""
    ridge_metrics = evaluate_model(y_test, ridge_preds)
    rf_metrics = evaluate_model(y_test, rf_preds)
    xgb_metrics = evaluate_model(y_test, xgb_preds)

    comparison = pd.DataFrame(
        [
            {
                "Model": "Ridge Regression",
                "Test MAE": ridge_metrics["MAE"],
                "Test RMSE": ridge_metrics["RMSE"],
                "Test R2": ridge_metrics["R2"],
                "Best 4-fold CV RMSE": -ridge_search.best_score_,
            },
            {
                "Model": "Random Forest",
                "Test MAE": rf_metrics["MAE"],
                "Test RMSE": rf_metrics["RMSE"],
                "Test R2": rf_metrics["R2"],
                "Best 4-fold CV RMSE": -rf_search.best_score_,
            },
            {
                "Model": "XGBoost",
                "Test MAE": xgb_metrics["MAE"],
                "Test RMSE": xgb_metrics["RMSE"],
                "Test R2": xgb_metrics["R2"],
                "Best 4-fold CV RMSE": -xgb_search.best_score_,
            },
        ]
    )
    comparison.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)
    return comparison


def save_performance_plot(comparison):
    """Save a simple bar chart comparing test RMSE."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(comparison["Model"], comparison["Test RMSE"], color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Test RMSE")
    ax.set_title("Final Test RMSE Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = RESULTS_DIR / "model_rmse_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_advanced_plots(
    best_ridge,
    best_rf,
    best_xgb,
    ridge_preds,
    rf_preds,
    xgb_preds,
    y_test,
    feature_names,
):
    """Save Actual vs Predicted plots and Feature Importance comparison."""
    print("Generating advanced model comparison plots...")

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Plot Ridge
    ax1.scatter(y_test, ridge_preds, alpha=0.3, color="#4C78A8")
    min_val_r = min(y_test.min(), ridge_preds.min())
    max_val_r = max(y_test.max(), ridge_preds.max())
    ax1.plot(
        [min_val_r, max_val_r], [min_val_r, max_val_r], "r--", lw=2, label="Perfect"
    )
    ax1.set_xlabel("Real Price (Target)")
    ax1.set_ylabel("Predicted Price (Model)")
    ax1.set_title("Ridge Regression")
    ax1.legend()

    # Plot Random Forest
    ax2.scatter(y_test, rf_preds, alpha=0.3, color="#F58518")
    min_val_rf = min(y_test.min(), rf_preds.min())
    max_val_rf = max(y_test.max(), rf_preds.max())
    ax2.plot(
        [min_val_rf, max_val_rf],
        [min_val_rf, max_val_rf],
        "r--",
        lw=2,
        label="Perfect",
    )
    ax2.set_xlabel("Real Price (Target)")
    ax2.set_title("Random Forest")
    ax2.legend()

    # Plot XGBoost
    ax3.scatter(y_test, xgb_preds, alpha=0.3, color="#2CA02C")
    min_val_xgb = min(y_test.min(), xgb_preds.min())
    max_val_xgb = max(y_test.max(), xgb_preds.max())
    ax3.plot(
        [min_val_xgb, max_val_xgb],
        [min_val_xgb, max_val_xgb],
        "r--",
        lw=2,
        label="Perfect",
    )
    ax3.set_xlabel("Real Price (Target)")
    ax3.set_title("XGBoost")
    ax3.legend()

    fig1.tight_layout()
    fig1.savefig(RESULTS_DIR / "actual_vs_predicted_comparison.png", dpi=300)
    plt.close(fig1)

    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(18, 6))

    # Importance Ridge
    ridge_coefs = best_ridge.named_steps["model"].coef_
    coef_series = pd.Series(ridge_coefs, index=feature_names).sort_values()
    ax4.barh(coef_series.index, coef_series.values, color="#4C78A8")
    ax4.set_title("Ridge: Coefficients")
    ax4.set_xlabel("Coefficient Value")

    # Importance Random Forest
    rf_importances = best_rf.feature_importances_
    imp_series_rf = pd.Series(rf_importances, index=feature_names).sort_values()
    ax5.barh(imp_series_rf.index, imp_series_rf.values, color="#F58518")
    ax5.set_title("Random Forest: Feature Importance")
    ax5.set_xlabel("Importance Level")

    # Importance XGBoost
    xgb_importances = best_xgb.feature_importances_
    imp_series_xgb = pd.Series(xgb_importances, index=feature_names).sort_values()
    ax6.barh(imp_series_xgb.index, imp_series_xgb.values, color="#2CA02C")
    ax6.set_title("XGBoost: Feature Importance")
    ax6.set_xlabel("Importance Level")

    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "feature_importance_comparison.png", dpi=300)
    plt.close(fig2)


def main():
    """Run the full project workflow."""
    create_directories()

    df, feature_names = load_dataset()
    analyze_features(
        df,
        feature_names,
    )
    x_train, x_test, y_train, y_test = save_train_test_data(df, feature_names)

    ridge_search, rf_search, xgb_search = tune_and_train_models(
        x_train,
        y_train,
    )

    best_ridge = ridge_search.best_estimator_
    best_rf = rf_search.best_estimator_
    best_xgb = xgb_search.best_estimator_

    print("Calculating final predictions on test set...")
    ridge_preds = best_ridge.predict(x_test)
    rf_preds = best_rf.predict(x_test)
    xgb_preds = best_xgb.predict(x_test)

    comparison = create_comparison_table(
        ridge_search,
        rf_search,
        xgb_search,
        ridge_preds,
        rf_preds,
        xgb_preds,
        y_test,
    )
    save_performance_plot(comparison)

    save_advanced_plots(
        best_ridge,
        best_rf,
        best_xgb,
        ridge_preds,
        rf_preds,
        xgb_preds,
        y_test,
        feature_names,
    )

    print("Lab 4 project completed successfully.")
    print("Result files saved to the Results and Models folders.")
    print("\nFinal model comparison:")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
