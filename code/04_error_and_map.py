from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score


def subgroup_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val, grp in df.groupby(group_col, dropna=False, observed=False):
        y_t = grp["target"].astype(int)
        y_p = grp["hgb_test_pred_tuned"].astype(int)
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        n = len(grp)
        rows.append(
            {
                "group_col": group_col,
                "group": str(val),
                "n_tracts": n,
                "pos_rate": round(float(y_t.mean()), 4),
                "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
                "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
                "f1": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
                "false_pos": fp,
                "false_neg": fn,
                "error_rate": round(float((fp + fn) / n), 4) if n > 0 else np.nan,
                "tp": tp,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(processed_dir / "test.csv", dtype={"CensusTract20": str})
    preds_df = pd.read_csv(processed_dir / "improved_test_predictions.csv", dtype={"CensusTract20": str})

    merged = pd.merge(
        preds_df,
        test_df[["CensusTract20", "PovertyRate", "MedianFamilyIncome", "TractHUNV", "OHU2020"]],
        on="CensusTract20",
        how="left",
    )

    merged["urban_label"] = np.where(merged["Urban"] == 1, "Urban (1)", "Rural (0)")
    merged["poverty_tier"] = pd.qcut(
        merged["PovertyRate"],
        q=3,
        labels=["Low Poverty (Tertile 1)", "Mid Poverty (Tertile 2)", "High Poverty (Tertile 3)"],
    )

    urban_tbl = subgroup_metrics(merged, "urban_label")
    pov_tbl = subgroup_metrics(merged, "poverty_tier")
    state_tbl = subgroup_metrics(merged, "State").sort_values("n_tracts", ascending=False).head(10)

    error_summary = pd.concat([urban_tbl, pov_tbl, state_tbl], ignore_index=True)
    error_summary.to_csv(metrics_dir / "error_analysis_subgroups.csv", index=False)

    # Static error-analysis chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.barplot(data=urban_tbl, x="group", y="recall", ax=axes[0])
    axes[0].set_title("Tuned Model Recall: Urban vs. Rural (Test)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.0)

    err_melt = urban_tbl.melt(
        id_vars="group",
        value_vars=["false_pos", "false_neg"],
        var_name="error_type",
        value_name="count",
    )
    sns.barplot(data=err_melt, x="group", y="count", hue="error_type", ax=axes[1])
    axes[1].set_title("False Positives vs. False Negatives: Urban vs. Rural")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Number of Test Tracts")

    plt.tight_layout()
    plt.savefig(figures_dir / "04_error_analysis_subgroups.png", dpi=150)
    plt.close()

    # Interactive Folium Choropleth Map (clean canvas without blocked external tile servers)
    geo_candidates = list((project_root / "data").rglob("*.geojson")) + list((project_root / "data").rglob("*.shp"))
    try:
        import folium
        import geopandas as gpd

        if geo_candidates:
            gdf = gpd.read_file(geo_candidates[0]).to_crs(epsg=4326)
            gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.001, preserve_topology=True)

            geoid_col = next((c for c in ["GEOID", "GEOID20", "CensusTract20"] if c in gdf.columns), None)
            if geoid_col is not None:
                gdf[geoid_col] = gdf[geoid_col].astype(str).str.strip().str.zfill(11)
                map_df = gdf.merge(
                    merged[
                        [
                            "CensusTract20",
                            "State",
                            "County24",
                            "PovertyRate",
                            "hgb_test_proba",
                            "target",
                            "hgb_test_pred_tuned",
                        ]
                    ],
                    left_on=geoid_col,
                    right_on="CensusTract20",
                    how="inner",
                )
                if not map_df.empty:
                    map_df["hgb_test_proba"] = map_df["hgb_test_proba"].round(4)
                    proj_centroids = map_df.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
                    center_lat = float(proj_centroids.y.mean())
                    center_lon = float(proj_centroids.x.mean())

                    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None)
                    m.get_root().html.add_child(
                        folium.Element("<style>.leaflet-container { background: #f8fafc; }</style>")
                    )

                    # Base outline layer of all tracts in the boundary file (full Texas silhouette)
                    folium.GeoJson(
                        gdf.to_json(),
                        name="All Tracts Outline",
                        style_function=lambda _: {
                            "fillColor": "#e2e8f0",
                            "color": "#64748b",
                            "weight": 0.45,
                            "fillOpacity": 0.35,
                        },
                    ).add_to(m)

                    choropleth = folium.Choropleth(
                        geo_data=map_df.to_json(),
                        data=map_df,
                        columns=["CensusTract20", "hgb_test_proba"],
                        key_on="feature.properties.CensusTract20",
                        fill_color="YlOrRd",
                        fill_opacity=0.82,
                        line_opacity=0.35,
                        nan_fill_opacity=0.0,
                        legend_name="Predicted Food Access Risk Probability (Held-Out Test Tracts)",
                    ).add_to(m)

                    folium.GeoJsonTooltip(
                        fields=["CensusTract20", "County24", "PovertyRate", "hgb_test_proba", "target", "hgb_test_pred_tuned"],
                        aliases=["Tract:", "County:", "Poverty Rate (%):", "Predicted Risk:", "Actual LILA (1&10):", "Tuned Pred:"],
                        localize=True,
                    ).add_to(choropleth.geojson)

                    folium.LayerControl().add_to(m)

                    out_html = figures_dir / "04_food_access_risk_map.html"
                    m.save(str(out_html))
                    print(f"Saved interactive map to: {out_html} ({len(map_df):,} test tracts mapped)")
            else:
                print("Geo file found, but no GEOID/GEOID20/CensusTract20 column matched.")
        else:
            print("No .geojson or .shp boundary file found under data/; skipped Folium map generation.")
    except ImportError:
        print("folium/geopandas not installed; skipped Folium map generation (subplot chart saved).")

    print("\n=== Urban vs. Rural Error Analysis ===")
    print(urban_tbl.to_string(index=False))
    print("\n=== Poverty Tier Error Analysis ===")
    print(pov_tbl.to_string(index=False))
    print(f"\nSaved error analysis table to: {metrics_dir / 'error_analysis_subgroups.csv'}")


if __name__ == "__main__":
    main()