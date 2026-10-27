"""
train.py - End-to-end training pipeline.
Runs preprocessing, HDBSCAN experiments, algorithm comparison,
stability analysis, and saves all results.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project dir to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from preprocessing import find_dataset, prepare_data, load_and_clean, get_dataset_info
from clustering import (
    apply_pca, run_hdbscan, run_kmeans, run_dbscan, run_agglomerative,
    hdbscan_grid_search, select_best_hdbscan, algorithm_comparison, compute_metrics,
    HDBSCAN_SOURCE
)
from evaluation import cluster_stability, profile_clusters, name_clusters, generate_recommendations

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42


def save_json(data, filename):
    """Save dict/list to JSON, handling numpy types."""
    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)
    print(f"  -> Saved {filename}")


def main():
    print("=" * 70)
    print("  CUSTOMER SEGMENTATION PIPELINE")
    print(f"  HDBSCAN source: {HDBSCAN_SOURCE}")
    print("=" * 70)

    # -- 1. Find and load dataset ------------------------------------------
    csv_path = find_dataset(str(PROJECT_DIR))
    if csv_path is None:
        print("\n[X] Dataset not found!")
        print("   Please download the IBM Telco Customer Churn dataset and place the CSV in:")
        print(f"   {PROJECT_DIR / 'data' / ''}")
        print("   Supported sources:")
        print("   • https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset")
        print("   • https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)

    print(f"\n[1] Loading dataset: {csv_path}")
    df_raw, X_scaled, feature_names, scaler, dataset_info = prepare_data(csv_path, include_churn_score=False)

    print(f"\n    Dataset shape: {dataset_info['shape']}")
    missing_cols = {k: v for k, v in dataset_info["missing"].items() if v > 0}
    if missing_cols:
        print(f"    Columns with missing values (after cleaning): {missing_cols}")
    print(f"    Features used for clustering: {len(feature_names)}")

    save_json({"feature_names": feature_names, "n_features": len(feature_names),
               "dataset_shape": list(dataset_info["shape"]),
               "csv_path": csv_path}, "dataset_info.json")

    # -- 2. Experiment A: Direct HDBSCAN ----------------------------------
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Standardized Features -> HDBSCAN")
    print("=" * 70)

    print("\n[2a] HDBSCAN hyperparameter grid search (no PCA)...")
    grid_results_A = hdbscan_grid_search(X_scaled)

    best_A = select_best_hdbscan(grid_results_A)
    if best_A is None:
        print("  [!] No valid HDBSCAN result found in Experiment A. Using default params.")
        labels_A, metrics_A, _ = run_hdbscan(X_scaled, min_cluster_size=20, min_samples=10)
        best_A = metrics_A
        best_A["labels"] = labels_A
    else:
        labels_A = best_A["labels"]
        print(f"\n  Best A: mcs={best_A['min_cluster_size']}, ms={best_A['min_samples']}, "
              f"clusters={best_A['n_clusters']}, noise={best_A['noise_pct']:.1f}%, "
              f"sil={best_A.get('silhouette', 'N/A')}")

    # -- 3. Experiment B: PCA -> HDBSCAN -----------------------------------
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Standardized Features -> PCA (95%) -> HDBSCAN")
    print("=" * 70)

    X_pca, pca_model = apply_pca(X_scaled, variance_threshold=0.95)

    print("\n[3b] HDBSCAN hyperparameter grid search (with PCA)...")
    grid_results_B = hdbscan_grid_search(X_pca)

    best_B = select_best_hdbscan(grid_results_B)
    if best_B is None:
        print("  [!] No valid HDBSCAN result in Experiment B. Using default params.")
        labels_B, metrics_B, _ = run_hdbscan(X_pca, min_cluster_size=20, min_samples=10)
        best_B = metrics_B
        best_B["labels"] = labels_B
    else:
        labels_B = best_B["labels"]
        print(f"\n  Best B: mcs={best_B['min_cluster_size']}, ms={best_B['min_samples']}, "
              f"clusters={best_B['n_clusters']}, noise={best_B['noise_pct']:.1f}%, "
              f"sil={best_B.get('silhouette', 'N/A')}")

    # Choose best overall experiment
    sil_A = best_A.get("silhouette") or -1
    sil_B = best_B.get("silhouette") or -1
    if sil_B > sil_A and best_B["n_clusters"] >= 2:
        best_overall = best_B
        best_experiment = "B (PCA + HDBSCAN)"
        best_labels = labels_B
        X_for_eval = X_pca
        print(f"\n  [OK] Best overall: Experiment B (PCA + HDBSCAN)")
    else:
        best_overall = best_A
        best_experiment = "A (Direct HDBSCAN)"
        best_labels = labels_A
        X_for_eval = X_scaled
        print(f"\n  [OK] Best overall: Experiment A (Direct HDBSCAN)")

    # -- 3.5. Optional: Churn Score experiment ----------------------------
    churn_score_experiment = None
    if "ChurnScore" in df_raw.columns:
        print("\n[3.5] Running experiment with ChurnScore as feature (for comparison only)...")
        _, X_cs, fn_cs, _, _ = prepare_data(csv_path, include_churn_score=True)
        labels_cs, metrics_cs, _ = run_hdbscan(X_cs,
                                                 min_cluster_size=best_overall.get("min_cluster_size", 20),
                                                 min_samples=best_overall.get("min_samples", 10))
        churn_score_experiment = {
            "with_churn_score": {k: v for k, v in metrics_cs.items() if k != "labels"},
            "without_churn_score": {k: v for k, v in best_overall.items() if k != "labels"},
            "note": "ChurnScore is outcome-derived. Including it risks target leakage. "
                    "Shown for comparison only; primary results exclude it."
        }
        save_json(churn_score_experiment, "churn_score_experiment.json")
        print(f"    With ChurnScore: {metrics_cs['n_clusters']} clusters, sil={metrics_cs.get('silhouette', 'N/A')}")
        print(f"    Without ChurnScore: {best_overall['n_clusters']} clusters, sil={best_overall.get('silhouette', 'N/A')}")

    # -- 4. Algorithm comparison ------------------------------------------
    print("\n" + "=" * 70)
    print("  ALGORITHM COMPARISON")
    print("=" * 70)

    best_for_comp = {k: v for k, v in best_overall.items() if k != "labels"}
    comparisons = algorithm_comparison(X_for_eval, best_labels, best_for_comp)

    for c in comparisons:
        print(f"  {c['algorithm']}: {c['n_clusters']} clusters, "
              f"noise={c.get('noise_pct', 0):.1f}%, "
              f"sil={c.get('silhouette', 'N/A')}")

    save_json(comparisons, "algorithm_comparison.json")

    # -- 5. Cluster stability ---------------------------------------------
    print("\n" + "=" * 70)
    print("  CLUSTER STABILITY ANALYSIS")
    print("=" * 70)

    from clustering import run_hdbscan as _run_hdbscan
    stability = cluster_stability(
        X_for_eval, _run_hdbscan, best_labels,
        min_cluster_size=best_overall.get("min_cluster_size", 20),
        min_samples=best_overall.get("min_samples", 10),
        n_iterations=10,
        subsample_frac=0.8,
        base_seed=RANDOM_SEED
    )
    print(f"  ARI: {stability['ari_mean']:.4f} ± {stability['ari_std']:.4f} "
          f"({stability['n_valid_runs']}/{stability['n_iterations']} valid runs)")
    print(f"  Noise handling: {stability['noise_handling']}")
    save_json(stability, "stability.json")

    # -- 6. Cluster profiling ---------------------------------------------
    print("\n" + "=" * 70)
    print("  CLUSTER PROFILING & BUSINESS ANALYSIS")
    print("=" * 70)

    profiles = profile_clusters(df_raw, best_labels)
    profiles = name_clusters(profiles)

    print(f"\n  Discovered {len(profiles)} segments (including noise if present):")
    for _, row in profiles.iterrows():
        churn_str = f", churn={row['Churn_Rate']:.1f}%" if "Churn_Rate" in row and pd.notna(row["Churn_Rate"]) else ""
        print(f"    {row['Cluster']} ({row['Persona']}): n={row['Count']} ({row['Pct']:.1f}%)"
              f"{churn_str}")

    profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    print(f"  -> Saved cluster_profiles.csv")

    recommendations = generate_recommendations(profiles)
    save_json(recommendations, "recommendations.json")

    # -- 7. Save all results ----------------------------------------------
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    # Save grid search results (without labels arrays)
    def clean_result(r):
        return {k: v for k, v in r.items() if k != "labels"}

    save_json([clean_result(r) for r in grid_results_A], "grid_results_A.json")
    save_json([clean_result(r) for r in grid_results_B], "grid_results_B.json")
    save_json(clean_result(best_overall), "best_model.json")
    save_json({"experiment": best_experiment}, "best_experiment.json")

    # Save customer assignments
    assignments = df_raw.copy()
    assignments["Cluster"] = best_labels
    if "CustomerID" in assignments.columns:
        out_cols = ["CustomerID", "Cluster"]
    else:
        out_cols = ["Cluster"]
    # Add key features
    for col in ["Tenure", "MonthlyCharges", "TotalCharges", "CLTV", "ChurnBinary"]:
        if col in assignments.columns:
            out_cols.append(col)
    assignments[out_cols].to_csv(OUTPUT_DIR / "customer_assignments.csv", index=False)
    print(f"  -> Saved customer_assignments.csv")

    # Save PCA info
    pca_info = {
        "n_components": int(pca_model.n_components_),
        "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(sum(pca_model.explained_variance_ratio_)),
    }
    save_json(pca_info, "pca_info.json")

    # Save PCA coordinates for visualization
    X_pca_vis = apply_pca(X_scaled, variance_threshold=0.95)[0]
    pca_df = pd.DataFrame(X_pca_vis[:, :min(3, X_pca_vis.shape[1])],
                          columns=[f"PC{i+1}" for i in range(min(3, X_pca_vis.shape[1]))])
    pca_df["Cluster"] = best_labels
    pca_df.to_csv(OUTPUT_DIR / "pca_coordinates.csv", index=False)
    print(f"  -> Saved pca_coordinates.csv")

    print("\n" + "=" * 70)
    print("  [OK] PIPELINE COMPLETE")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
