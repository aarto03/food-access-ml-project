from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_COLS = [
    "Urban",
    "POP2020",
    "OHU2020",
    "NUMGQTRS",
    "PCTGQTRS",
    "PovertyRate",
    "MedianFamilyIncome",
    "TractLOWI",
    "TractKids",
    "TractSeniors",
    "TractWhite",
    "TractBlack",
    "TractAsian",
    "TractNHOPI",
    "TractAIAN",
    "TractOMultir",
    "TractHispanic",
    "TractHUNV",
    "TractSNAP",
    "TractVeteran",
    "TractTribalArea",
]


def compute_metrics(model_name: str, split_name: str, threshold: float, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": model_name,
        "split": split_name,
        "threshold": round(float(threshold), 4),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv", dtype={"CensusTract20": str})
    val_df = pd.read_csv(processed_dir / "val.csv", dtype={"CensusTract20": str})
    test_df = pd.read_csv(processed_dir / "test.csv", dtype={"CensusTract20": str})

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"].astype(int)
    X_val, y_val = val_df[FEATURE_COLS], val_df["target"].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"].astype(int)

    # Hyperparameter candidates evaluated strictly on the validation set
    param_grid = [
        {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
        {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
        {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 30},
    ]

    best_val_auc = -1.0
    best_model = None
    best_params = None

    for params in param_grid:
        clf = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=42,
            **params,
        )
        clf.fit(X_train, y_train)
        val_proba_cand = clf.predict_proba(X_val)[:, 1]
        val_auc_cand = roc_auc_score(y_val, val_proba_cand)
        print(f"Params {params} -> Val ROC-AUC: {val_auc_cand:.4f}")
        if val_auc_cand > best_val_auc:
            best_val_auc = val_auc_cand
            best_model = clf
            best_params = params

    assert best_model is not None and best_params is not None

    # Select decision threshold on validation set to maximize F1
    val_proba = best_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.20, 0.85, 66)
    best_threshold = 0.50
    best_val_f1 = -1.0

    for th in thresholds:
        val_pred_th = (val_proba >= th).astype(int)
        f1_th = f1_score(y_val, val_pred_th, zero_division=0)
        if f1_th > best_val_f1:
            best_val_f1 = f1_th
            best_threshold = float(th)

    # Evaluate on held-out test set using both default (0.50) and validation-tuned threshold
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred_default = (test_proba >= 0.50).astype(int)
    test_pred_tuned = (test_proba >= best_threshold).astype(int)

    comparison_rows = [
        compute_metrics("HistGB_Balanced_DefaultTh", "test", 0.50, y_test, test_pred_default, test_proba),
        compute_metrics("HistGB_Balanced_TunedTh", "test", best_threshold, y_test, test_pred_tuned, test_proba),
    ]

    baseline_path = metrics_dir / "baseline_metrics.csv"
    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)
        base_test = base_df[base_df["split"] == "test"].copy()
        if not base_test.empty:
            base_test["threshold"] = 0.50
            cols_order = [
                "model",
                "split",
                "threshold",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
            comparison_df = pd.concat([base_test[cols_order], pd.DataFrame(comparison_rows)[cols_order]], ignore_index=True)
        else:
            comparison_df = pd.DataFrame(comparison_rows)
    else:
        comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

    # Save improved test predictions for Phase 4 error analysis & mapping
    improved_preds_out = test_df[["CensusTract20", "State", "County20", "County24", "Urban", "target"]].copy()
    improved_preds_out["hgb_test_proba"] = test_proba
    improved_preds_out["hgb_test_pred_default"] = test_pred_default
    improved_preds_out["hgb_test_pred_tuned"] = test_pred_tuned
    improved_preds_out["tuned_threshold"] = best_threshold
    improved_preds_out.to_csv(processed_dir / "improved_test_predictions.csv", index=False)

    # Plot ROC comparison if baseline predictions are available
    fig, ax = plt.subplots(figsize=(7, 5.5))
    RocCurveDisplay.from_predictions(
        y_test,
        test_proba,
        name=f"HistGradientBoosting (AUC = {roc_auc_score(y_test, test_proba):.4f})",
        ax=ax,
    )

    base_preds_path = processed_dir / "baseline_test_predictions.csv"
    if base_preds_path.exists():
        base_preds = pd.read_csv(base_preds_path)
        if "lr_test_proba" in base_preds.columns:
            RocCurveDisplay.from_predictions(
                y_test,
                base_preds["lr_test_proba"],
                name=f"Logistic Regression Baseline (AUC = {roc_auc_score(y_test, base_preds['lr_test_proba']):.4f})",
                ax=ax,
            )

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)")
    ax.set_title("Model Comparison: Held-Out Test Set ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_model_comparison_roc.png", dpi=150)
    plt.close()

    print(f"\nBest Validation Params: {best_params} (Val ROC-AUC: {best_val_auc:.4f})")
    print(f"Tuned Threshold (max Val F1): {best_threshold:.2f} (Val F1: {best_val_f1:.4f})\n")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison to: {metrics_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()