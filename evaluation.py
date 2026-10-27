"""
evaluation.py - Cluster stability, profiling, and business analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def cluster_stability(X: np.ndarray, run_hdbscan_fn, labels_full: np.ndarray,
                      min_cluster_size: int = 20, min_samples: int = 10,
                      n_iterations: int = 10, subsample_frac: float = 0.8,
                      base_seed: int = 42) -> dict:
    """
    Evaluate clustering stability via subsampling + ARI comparison.

    Method:
        - Sample `subsample_frac` of data `n_iterations` times.
        - Refit HDBSCAN on each subsample.
        - Compute ARI between subsample labels and full-data labels on shared indices.
        - Noise points (label -1) are INCLUDED with their own label for ARI computation.
          This is documented explicitly.

    Returns dict with ARI mean, std, individual scores, and metadata.
    """
    n = X.shape[0]
    n_sub = int(n * subsample_frac)
    ari_scores = []

    for i in range(n_iterations):
        rng = np.random.RandomState(base_seed + i)
        indices = rng.choice(n, size=n_sub, replace=False)
        X_sub = X[indices]

        try:
            sub_labels, _, _ = run_hdbscan_fn(X_sub, min_cluster_size=min_cluster_size,
                                               min_samples=min_samples)
        except Exception:
            continue

        # Compare with full-data labels on shared indices
        full_labels_subset = labels_full[indices]

        # ARI including noise label (-1)
        ari = adjusted_rand_score(full_labels_subset, sub_labels)
        ari_scores.append(round(float(ari), 4))

    result = {
        "n_iterations": n_iterations,
        "n_valid_runs": len(ari_scores),
        "subsample_frac": subsample_frac,
        "ari_scores": ari_scores,
        "ari_mean": round(float(np.mean(ari_scores)), 4) if ari_scores else None,
        "ari_std": round(float(np.std(ari_scores)), 4) if ari_scores else None,
        "noise_handling": "Noise label (-1) included in ARI computation",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }

    return result


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Profile each cluster using original customer attributes.
    Label -1 is treated as Noise/Outliers.
    """
    df_work = df.copy()
    df_work["Cluster"] = labels

    profiles = []
    for cl in sorted(df_work["Cluster"].unique()):
        subset = df_work[df_work["Cluster"] == cl]
        n = len(subset)

        profile = {
            "Cluster": f"Noise/Outliers" if cl == -1 else f"Cluster {cl}",
            "ClusterID": int(cl),
            "Count": n,
            "Pct": round(n / len(df_work) * 100, 1),
        }

        if "Tenure" in df_work.columns:
            profile["Mean_Tenure"] = round(float(subset["Tenure"].mean()), 1)
            profile["Median_Tenure"] = round(float(subset["Tenure"].median()), 1)

        if "MonthlyCharges" in df_work.columns:
            profile["Mean_MonthlyCharges"] = round(float(subset["MonthlyCharges"].mean()), 2)

        if "TotalCharges" in df_work.columns:
            profile["Mean_TotalCharges"] = round(float(subset["TotalCharges"].mean()), 2)

        if "CLTV" in df_work.columns:
            profile["Mean_CLTV"] = round(float(subset["CLTV"].mean()), 2)

        if "ChurnBinary" in df_work.columns:
            profile["Churn_Rate"] = round(float(subset["ChurnBinary"].mean()) * 100, 1)

        if "NumReferrals" in df_work.columns:
            profile["Mean_Referrals"] = round(float(subset["NumReferrals"].mean()), 1)

        if "Contract" in df_work.columns:
            profile["Top_Contract"] = subset["Contract"].mode().iloc[0] if len(subset["Contract"].mode()) > 0 else "N/A"

        if "InternetService" in df_work.columns:
            profile["Top_InternetService"] = subset["InternetService"].mode().iloc[0] if len(subset["InternetService"].mode()) > 0 else "N/A"

        if "PaymentMethod" in df_work.columns:
            profile["Top_PaymentMethod"] = subset["PaymentMethod"].mode().iloc[0] if len(subset["PaymentMethod"].mode()) > 0 else "N/A"

        profiles.append(profile)

    return pd.DataFrame(profiles)


def name_clusters(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign descriptive names to clusters based on observed characteristics.
    Names are derived from actual data, not assumed.
    """
    names = []
    for _, row in profile_df.iterrows():
        if row["ClusterID"] == -1:
            names.append("Noise / Outliers")
            continue

        parts = []

        # Tenure-based
        if "Mean_Tenure" in row:
            if row["Mean_Tenure"] < 12:
                parts.append("New")
            elif row["Mean_Tenure"] < 36:
                parts.append("Mid-tenure")
            else:
                parts.append("Long-tenure")

        # Value-based
        if "Mean_MonthlyCharges" in row:
            if row["Mean_MonthlyCharges"] < 40:
                parts.append("Low-spend")
            elif row["Mean_MonthlyCharges"] < 75:
                parts.append("Mid-spend")
            else:
                parts.append("High-spend")

        # Churn-based
        if "Churn_Rate" in row:
            if row["Churn_Rate"] > 40:
                parts.append("High-churn")
            elif row["Churn_Rate"] < 15:
                parts.append("Loyal")

        if parts:
            names.append(" / ".join(parts))
        else:
            names.append(f"Segment {row['ClusterID']}")

    profile_df = profile_df.copy()
    profile_df["Persona"] = names
    return profile_df


def generate_recommendations(profile_df: pd.DataFrame) -> list[dict]:
    """Generate cautious business recommendations based on cluster profiles."""
    recs = []
    for _, row in profile_df.iterrows():
        if row["ClusterID"] == -1:
            recs.append({
                "Cluster": row["Cluster"],
                "Persona": row.get("Persona", "Noise"),
                "Recommendation": "These customers don't fit well into any segment. "
                                  "Investigate individually — they may represent data quality "
                                  "issues or truly unique customer profiles."
            })
            continue

        rec_parts = []

        if "Churn_Rate" in row and row["Churn_Rate"] > 40:
            rec_parts.append("High observed churn rate suggests retention interventions "
                           "may be worthwhile for this segment.")

        if "Mean_Tenure" in row and row["Mean_Tenure"] < 12:
            rec_parts.append("Relatively new customers — onboarding experience and "
                           "early engagement are key.")

        if "Mean_MonthlyCharges" in row and row["Mean_MonthlyCharges"] > 75:
            rec_parts.append("High-value segment — prioritize service quality and "
                           "personalized retention offers.")

        if "Mean_Tenure" in row and row["Mean_Tenure"] > 36 and \
           "Churn_Rate" in row and row["Churn_Rate"] < 20:
            rec_parts.append("Loyal, long-tenured customers — consider referral programs "
                           "and loyalty rewards.")

        if "Mean_MonthlyCharges" in row and row["Mean_MonthlyCharges"] < 40:
            rec_parts.append("Low-spend segment — explore upselling opportunities "
                           "with relevant add-on services.")

        if not rec_parts:
            rec_parts.append("Standard customer segment. Monitor KPIs and maintain "
                           "consistent service quality.")

        recs.append({
            "Cluster": row["Cluster"],
            "Persona": row.get("Persona", f"Segment {row['ClusterID']}"),
            "Recommendation": " ".join(rec_parts)
        })

    return recs
