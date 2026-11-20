from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def eval_split(split_name: str, model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": model_name,
        "split": split_name,
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

    rows = []

    # 1. Dummy majority-class baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    rows.append(
        eval_split(
            "test",
            "Dummy_MostFrequent",
            y_test,
            dummy.predict(X_test),
            dummy.predict_proba(X_test)[:, 1],
        )
    )

    # 2. Logistic Regression baseline (with feature standardization)
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    lr_pipe.fit(X_train, y_train)

    val_proba = lr_pipe.predict_proba(X_val)[:, 1]
    val_pred = lr_pipe.predict(X_val)
    rows.append(eval_split("val", "LogisticRegression", y_val, val_pred, val_proba))

    test_proba = lr_pipe.predict_proba(X_test)[:, 1]
    test_pred = lr_pipe.predict(X_test)
    rows.append(eval_split("test", "LogisticRegression", y_test, test_pred, test_proba))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_dir / "baseline_metrics.csv", index=False)

    # Save test predictions for downstream comparison & error analysis
    test_preds_out = test_df[["CensusTract20", "State", "County20", "County24", "Urban", "target"]].copy()
    test_preds_out["lr_test_proba"] = test_proba
    test_preds_out["lr_test_pred"] = test_pred
    test_preds_out.to_csv(processed_dir / "baseline_test_predictions.csv", index=False)

    # ROC curve on test set
    fig, ax = plt.subplots(figsize=(6.5, 5))
    RocCurveDisplay.from_predictions(y_test, test_proba, name="Logistic Regression (Baseline)", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    ax.set_title("Baseline ROC Curve (Held-Out Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_baseline_roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix on test set
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_pred,
        display_labels=["Adequate (0)", "High Risk (1)"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Baseline Confusion Matrix (Test Set, Threshold = 0.50)")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_baseline_confusion_matrix.png", dpi=150)
    plt.close()

    print(metrics_df.to_string(index=False))
    print(f"Saved metrics to: {metrics_dir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()