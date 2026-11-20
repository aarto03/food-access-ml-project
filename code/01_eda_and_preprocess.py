from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "2025-sram-fara-data"
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    gen_path = raw_dir / "SRAM General Tract Characteristics Data.csv"
    dd_path = raw_dir / "SRAM Driving Distance Data.csv"

    # SRAM CSVs use latin1 encoding
    gen_df = pd.read_csv(gen_path, encoding="latin1", low_memory=False)
    dd_df = pd.read_csv(dd_path, encoding="latin1", low_memory=False)

    gen_df["CensusTract20"] = gen_df["CensusTract20"].astype(str).str.strip().str.zfill(11)
    dd_df["CensusTract20"] = dd_df["CensusTract20"].astype(str).str.strip().str.zfill(11)

    target_raw = "DD_SRAM_LILATracts_1And10"
    df = pd.merge(
        gen_df,
        dd_df[["CensusTract20", target_raw]],
        on="CensusTract20",
        how="inner",
    )

    df = df.dropna(subset=[target_raw]).copy()
    df["target"] = df[target_raw].astype(int)

    feature_cols = [
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
    meta_cols = ["CensusTract20", "State", "County20", "County24"]

    # Save EDA summary statistics before imputation
    eda_summary = df[feature_cols + ["target"]].describe().T
    eda_summary["missing_count"] = df[feature_cols + ["target"]].isna().sum()
    eda_summary.to_csv(metrics_dir / "eda_summary_stats.csv")

    # EDA chart: Target balance by Urban vs. Rural
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="Urban", hue="target")
    plt.title("Target Class Distribution by Urban Flag (0=Rural, 1=Urban)")
    plt.xlabel("Urban Flag")
    plt.ylabel("Number of Census Tracts")
    plt.legend(title="LILA 1 & 10 (target)")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_target_by_urban.png", dpi=150)
    plt.close()

    # EDA chart: PovertyRate distribution by target
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="target", y="PovertyRate")
    plt.title("Tract Poverty Rate by Food Access Target")
    plt.xlabel("LILA 1 & 10 Target (0 = No, 1 = Yes)")
    plt.ylabel("Poverty Rate (%)")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_poverty_by_target.png", dpi=150)
    plt.close()

    X = df[feature_cols].copy()
    y = df["target"].copy()
    meta = df[meta_cols].copy()

    # Stratified split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X, y, meta, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Fit median imputation strictly on training split to avoid leakage
    train_medians = X_train.median()
    X_train_clean = X_train.fillna(train_medians)
    X_val_clean = X_val.fillna(train_medians)
    X_test_clean = X_test.fillna(train_medians)

    # Save splits for reproducible downstream training and evaluation
    for split_name, X_s, y_s, m_s in [
        ("train", X_train_clean, y_train, meta_train),
        ("val", X_val_clean, y_val, meta_val),
        ("test", X_test_clean, y_test, meta_test),
    ]:
        out_df = pd.concat([m_s.reset_index(drop=True), X_s.reset_index(drop=True), y_s.reset_index(drop=True)], axis=1)
        out_df.to_csv(processed_dir / f"{split_name}.csv", index=False)

    print(f"Usable tracts: {len(df):,} | Positive rate: {y.mean():.4f}")
    print(f"Splits saved to: {processed_dir}")
    print(f"Train: {len(X_train_clean):,} | Val: {len(X_val_clean):,} | Test: {len(X_test_clean):,}")


if __name__ == "__main__":
    main()