"""
clustering.py - HDBSCAN, K-Means, DBSCAN clustering + PCA + hyperparameter tuning.
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

warnings.filterwarnings("ignore")

# Try sklearn HDBSCAN first (available since sklearn 1.3+)
try:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_SOURCE = "sklearn"
except ImportError:
    try:
        from hdbscan import HDBSCAN
        HDBSCAN_SOURCE = "hdbscan"
    except ImportError:
        HDBSCAN = None
        HDBSCAN_SOURCE = None


def apply_pca(X_scaled: np.ndarray, variance_threshold: float = 0.95) -> tuple[np.ndarray, PCA]:
    """Apply PCA retaining given variance. Handles low-dimensional data safely."""
    n_features = X_scaled.shape[1]
    n_samples = X_scaled.shape[0]
    max_components = min(n_features, n_samples)

    pca = PCA(n_components=min(max_components, n_features))
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = max(2, min(n_components, max_components))

    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_scaled)

    print(f"[PCA] {n_features} features → {n_components} components ({cumvar[n_components-1]*100:.1f}% variance)")
    return X_pca, pca_final


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering metrics, handling edge cases."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    noise_pct = n_noise / len(labels) * 100

    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(noise_pct, 2),
        "n_samples": len(labels),
    }

    # Cluster size distribution
    from collections import Counter
    size_dist = Counter(labels)
    metrics["cluster_sizes"] = {int(k): v for k, v in sorted(size_dist.items())}

    # Non-noise mask
    non_noise = labels != -1
    n_non_noise = int(non_noise.sum())

    if n_clusters >= 2 and n_non_noise >= 2:
        X_nn = X[non_noise]
        l_nn = labels[non_noise]

        # Check that we have at least 2 unique labels in non-noise
        if len(set(l_nn)) >= 2:
            try:
                metrics["silhouette"] = round(float(silhouette_score(X_nn, l_nn)), 4)
            except Exception:
                metrics["silhouette"] = None

            try:
                metrics["davies_bouldin"] = round(float(davies_bouldin_score(X_nn, l_nn)), 4)
            except Exception:
                metrics["davies_bouldin"] = None

            try:
                metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X_nn, l_nn)), 4)
            except Exception:
                metrics["calinski_harabasz"] = None
        else:
            metrics["silhouette"] = None
            metrics["davies_bouldin"] = None
            metrics["calinski_harabasz"] = None
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
        if n_clusters < 2:
            metrics["note"] = "Fewer than 2 clusters found; metrics not computable."

    return metrics


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 20, min_samples: int = 10) -> tuple[np.ndarray, dict, float]:
    """Run HDBSCAN and return labels, metrics, runtime."""
    if HDBSCAN is None:
        raise ImportError("HDBSCAN not available. Install scikit-learn>=1.3 or the hdbscan package.")

    start = time.time()
    if HDBSCAN_SOURCE == "sklearn":
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        store_centers="centroid")
    else:
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    labels = model.fit_predict(X)
    elapsed = time.time() - start

    metrics = compute_metrics(X, labels)
    metrics["min_cluster_size"] = min_cluster_size
    metrics["min_samples"] = min_samples
    metrics["runtime_s"] = round(elapsed, 3)
    metrics["algorithm"] = "HDBSCAN"

    return labels, metrics, elapsed


def run_kmeans(X: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> tuple[np.ndarray, dict, float]:
    """Run K-Means."""
    start = time.time()
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = model.fit_predict(X)
    elapsed = time.time() - start

    metrics = compute_metrics(X, labels)
    metrics["n_clusters_param"] = n_clusters
    metrics["runtime_s"] = round(elapsed, 3)
    metrics["algorithm"] = "KMeans"

    return labels, metrics, elapsed


def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> tuple[np.ndarray, dict, float]:
    """Run DBSCAN."""
    start = time.time()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    elapsed = time.time() - start

    metrics = compute_metrics(X, labels)
    metrics["eps"] = eps
    metrics["min_samples"] = min_samples
    metrics["runtime_s"] = round(elapsed, 3)
    metrics["algorithm"] = "DBSCAN"

    return labels, metrics, elapsed


def run_agglomerative(X: np.ndarray, n_clusters: int = 5) -> tuple[np.ndarray, dict, float]:
    """Run Agglomerative Clustering."""
    start = time.time()
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    elapsed = time.time() - start

    metrics = compute_metrics(X, labels)
    metrics["n_clusters_param"] = n_clusters
    metrics["runtime_s"] = round(elapsed, 3)
    metrics["algorithm"] = "Agglomerative"

    return labels, metrics, elapsed


def hdbscan_grid_search(X: np.ndarray,
                        min_cluster_sizes: list[int] | None = None,
                        min_samples_list: list[int] | None = None) -> list[dict]:
    """
    Tune HDBSCAN over a grid of hyperparameters.
    Adapts grid to dataset size.
    """
    n = X.shape[0]

    if min_cluster_sizes is None:
        if n < 500:
            min_cluster_sizes = [5, 10, 15]
        elif n < 2000:
            min_cluster_sizes = [10, 20, 30]
        else:
            min_cluster_sizes = [10, 20, 30, 50]

    if min_samples_list is None:
        if n < 500:
            min_samples_list = [3, 5, 10]
        elif n < 2000:
            min_samples_list = [5, 10, 15]
        else:
            min_samples_list = [5, 10, 20]

    results = []
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            try:
                labels, metrics, elapsed = run_hdbscan(X, min_cluster_size=mcs, min_samples=ms)
                metrics["labels"] = labels
                results.append(metrics)
                print(f"  HDBSCAN(mcs={mcs}, ms={ms}): {metrics['n_clusters']} clusters, "
                      f"{metrics['noise_pct']:.1f}% noise, sil={metrics.get('silhouette', 'N/A')}, "
                      f"{elapsed:.2f}s")
            except Exception as e:
                print(f"  HDBSCAN(mcs={mcs}, ms={ms}): FAILED - {e}")
                results.append({
                    "min_cluster_size": mcs, "min_samples": ms,
                    "error": str(e), "algorithm": "HDBSCAN"
                })

    return results


def select_best_hdbscan(results: list[dict]) -> dict | None:
    """
    Select best HDBSCAN result balancing silhouette, noise %, and cluster count.
    
    Penalizes excessive noise (>50%) and excessive fragmentation (>30 clusters).
    Uses a composite score: silhouette * (1 - noise_penalty) * cluster_penalty.
    """
    valid = [r for r in results if r.get("silhouette") is not None and r["n_clusters"] >= 2]
    if not valid:
        # Fallback: pick result with reasonable cluster count and lowest noise
        valid = [r for r in results if r.get("n_clusters", 0) >= 2 and "labels" in r]
        if not valid:
            valid = [r for r in results if "labels" in r]
        if not valid:
            return None
        return min(valid, key=lambda r: r.get("noise_pct", 100))

    def composite_score(r):
        sil = r["silhouette"]
        noise = r.get("noise_pct", 0) / 100.0
        n_clust = r.get("n_clusters", 2)
        
        # Penalize high noise (prefer <30%)
        if noise > 0.5:
            noise_penalty = 0.5
        elif noise > 0.3:
            noise_penalty = 0.2
        else:
            noise_penalty = 0.0
        
        # Penalize excessive fragmentation (prefer 3-15 clusters)
        if n_clust > 30:
            cluster_penalty = 0.4
        elif n_clust > 15:
            cluster_penalty = 0.2
        elif n_clust < 3:
            cluster_penalty = 0.1
        else:
            cluster_penalty = 0.0
        
        return sil * (1 - noise_penalty) * (1 - cluster_penalty)
    
    return max(valid, key=composite_score)


def algorithm_comparison(X: np.ndarray, best_hdbscan_labels: np.ndarray | None = None,
                         best_hdbscan_metrics: dict | None = None) -> list[dict]:
    """Compare HDBSCAN vs K-Means vs DBSCAN vs Agglomerative."""
    comparisons = []

    # HDBSCAN (use best result)
    if best_hdbscan_metrics is not None:
        entry = {k: v for k, v in best_hdbscan_metrics.items() if k != "labels"}
        comparisons.append(entry)

    # K-Means with k matching HDBSCAN clusters, or default 4
    n_clusters_km = best_hdbscan_metrics["n_clusters"] if best_hdbscan_metrics and best_hdbscan_metrics.get("n_clusters", 0) >= 2 else 4
    n_clusters_km = max(2, min(n_clusters_km, 8))  # Cap at 8 for meaningful comparison
    _, km_metrics, _ = run_kmeans(X, n_clusters=n_clusters_km)
    comparisons.append(km_metrics)

    # DBSCAN — try a few eps values, pick best
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    eps_est = float(np.percentile(distances[:, -1], 90))
    for eps_mult in [0.8, 1.0, 1.2]:
        eps_val = round(eps_est * eps_mult, 3)
        _, db_metrics, _ = run_dbscan(X, eps=eps_val, min_samples=10)
        if db_metrics.get("n_clusters", 0) >= 2:
            comparisons.append(db_metrics)
            break
    else:
        _, db_metrics, _ = run_dbscan(X, eps=eps_est, min_samples=10)
        comparisons.append(db_metrics)

    # Agglomerative
    _, agg_metrics, _ = run_agglomerative(X, n_clusters=n_clusters_km)
    comparisons.append(agg_metrics)

    return comparisons
