"""
app.py - Streamlit Dashboard for Customer Segmentation Results.
Professional, interactive dashboard with Plotly visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR = PROJECT_DIR / "data"


def load_json(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_csv(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
/* Main application */
.stApp {
    background-color: #F5F7FB;
    color: #172B4D;
}

/* Main content area */
[data-testid="stMain"] {
    background-color: #F5F7FB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #17365D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #17365D !important;
    font-weight: 700;
}

/* General text */
.stApp p,
.stApp li,
.stApp label {
    color: #334155;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE5F0;
    border-left: 5px solid #0F9D8A;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(23, 54, 93, 0.07);
}

/* KPI labels */
div[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-size: 14px;
    font-weight: 600;
}

/* KPI values */
div[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-size: 30px;
    font-weight: 750;
}

/* KPI delta text */
div[data-testid="stMetricDelta"] {
    color: #0F766E !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #334155;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0F766E !important;
}

/* Tables and dataframes */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE5F0;
    border-radius: 10px;
}

/* Inputs and select boxes */
[data-testid="stTextInput"],
[data-testid="stSelectbox"],
[data-testid="stNumberInput"] {
    background-color: #FFFFFF;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Section dividers */
hr {
    border-color: #DCE5F0;
}

/* Header typography */
.main-header {
    font-size: 2.1rem;
    font-weight: 800;
    color: #17365D;
    margin-bottom: 0.5rem;
}

.sub-header {
    font-size: 1rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)



# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Controls")
    st.markdown("---")

    # Check if results exist
    results_exist = (OUTPUT_DIR / "best_model.json").exists()

    if not results_exist:
        st.warning("⚠️ No results found. Run `python train.py` first.")

        # Dataset upload
        st.markdown("### 📁 Upload Dataset")
        uploaded_file = st.file_uploader("Upload Telco Churn CSV", type=["csv"])
        if uploaded_file is not None:
            DATA_DIR.mkdir(exist_ok=True)
            save_path = DATA_DIR / "telco_churn.csv"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success(f"Saved to {save_path}")
            st.info("Now run `python train.py` in the project directory.")

        # Run training button
        if st.button("🚀 Run Training Pipeline"):
            import subprocess
            with st.spinner("Running training pipeline..."):
                result = subprocess.run(
                    ["python", str(PROJECT_DIR / "train.py")],
                    capture_output=True, text=True, cwd=str(PROJECT_DIR)
                )
                if result.returncode == 0:
                    st.success("✅ Training complete!")
                    st.rerun()
                else:
                    st.error("❌ Training failed:")
                    st.code(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])

    else:
        st.success("✅ Results loaded")

        # Download buttons
        st.markdown("### 📥 Downloads")
        assignments = load_csv("customer_assignments.csv")
        if assignments is not None:
            st.download_button(
                "📋 Customer Assignments",
                assignments.to_csv(index=False),
                "customer_assignments.csv",
                "text/csv"
            )

        profiles_df = load_csv("cluster_profiles.csv")
        if profiles_df is not None:
            st.download_button(
                "📊 Cluster Profiles",
                profiles_df.to_csv(index=False),
                "cluster_profiles.csv",
                "text/csv"
            )

        grid_a = load_json("grid_results_A.json")
        if grid_a:
            st.download_button(
                "🔬 Experiment Results (JSON)",
                json.dumps(grid_a, indent=2),
                "experiment_results.json",
                "application/json"
            )

    st.markdown("---")
    st.caption("Built with HDBSCAN, Scikit-learn & Streamlit")
    st.caption("Research Mini-Project 2026")


# ─── Main Content ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 Customer Segmentation & Value Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">HDBSCAN-Based Customer Persona Discovery | IBM Telco Churn Dataset</div>', unsafe_allow_html=True)

if not results_exist:
    st.info("👈 Upload a dataset and run the training pipeline using the sidebar controls.")
    st.markdown("""
    ### Getting Started
    1. **Download** the IBM Telco Customer Churn dataset from
       [Kaggle (IBM version)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
       or [Kaggle (original)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    2. **Upload** the CSV file using the sidebar
    3. **Run** the training pipeline
    4. **Explore** the interactive results
    """)
    st.stop()


# ─── Load all results ───────────────────────────────────────────────────
best_model = load_json("best_model.json")
best_experiment = load_json("best_experiment.json")
profiles_df = load_csv("cluster_profiles.csv")
pca_coords = load_csv("pca_coordinates.csv")
pca_info = load_json("pca_info.json")
dataset_info = load_json("dataset_info.json")
grid_A = load_json("grid_results_A.json")
grid_B = load_json("grid_results_B.json")
comparisons = load_json("algorithm_comparison.json")
stability = load_json("stability.json")
recommendations = load_json("recommendations.json")
churn_score_exp = load_json("churn_score_experiment.json")
assignments = load_csv("customer_assignments.csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔍 Clusters", "📈 Experiments",
    "🔄 Comparison", "📋 Stability", "💡 Insights"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview & Key Metrics")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_customers = dataset_info["dataset_shape"][0] if dataset_info else "N/A"
        st.metric("👥 Total Customers", f"{n_customers:,}" if isinstance(n_customers, int) else n_customers)
    with col2:
        n_clusters = best_model.get("n_clusters", "N/A") if best_model else "N/A"
        st.metric("🎯 Clusters Found", n_clusters)
    with col3:
        noise_pct = best_model.get("noise_pct", "N/A") if best_model else "N/A"
        st.metric("🔇 Noise %", f"{noise_pct}%" if isinstance(noise_pct, (int, float)) else noise_pct)
    with col4:
        sil = best_model.get("silhouette", "N/A") if best_model else "N/A"
        st.metric("📐 Silhouette Score", sil if sil is not None else "N/A")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📋 Dataset Info")
        if dataset_info:
            st.markdown(f"- **Shape:** {dataset_info['dataset_shape'][0]:,} rows × {dataset_info['dataset_shape'][1]} columns")
            st.markdown(f"- **Clustering Features:** {dataset_info['n_features']}")
            st.markdown(f"- **Best Experiment:** {best_experiment.get('experiment', 'N/A') if best_experiment else 'N/A'}")
            if best_model:
                st.markdown(f"- **HDBSCAN params:** min_cluster_size={best_model.get('min_cluster_size')}, "
                          f"min_samples={best_model.get('min_samples')}")
                st.markdown(f"- **Runtime:** {best_model.get('runtime_s', 'N/A')}s")

        st.subheader("🔢 Feature Set")
        if dataset_info:
            feat_names = dataset_info.get("feature_names", [])
            # Separate numeric and one-hot
            numeric_feats = [f for f in feat_names if "_" not in f or f in ["AvgMonthlyRevenue"]]
            categorical_feats = [f for f in feat_names if f not in numeric_feats]
            st.markdown(f"**Numeric ({len(numeric_feats)}):** {', '.join(numeric_feats)}")
            if categorical_feats:
                st.markdown(f"**One-Hot Encoded ({len(categorical_feats)}):** {len(categorical_feats)} binary features from categorical variables")

    with col_right:
        st.subheader("🔍 Data Quality")
        if dataset_info:
            missing = {k: v for k, v in dataset_info.get("missing", {}).items() if v > 0}
            if missing:
                missing_df = pd.DataFrame([
                    {"Column": k, "Missing": v, "Pct": dataset_info["missing_pct"].get(k, 0)}
                    for k, v in missing.items()
                ])
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("No missing values in the dataset! ✅")

        # PCA info
        if pca_info:
            st.subheader("📊 PCA Summary")
            st.markdown(f"- **Components retained:** {pca_info['n_components']}")
            st.markdown(f"- **Variance explained:** {pca_info['total_variance_explained']*100:.1f}%")

            fig_pca_var = go.Figure()
            cumvar = np.cumsum(pca_info["explained_variance_ratio"])
            fig_pca_var.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(pca_info["explained_variance_ratio"]))],
                y=pca_info["explained_variance_ratio"],
                name="Individual",
                marker_color="#667eea"
            ))
            fig_pca_var.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cumvar))],
                y=cumvar.tolist(),
                name="Cumulative",
                line=dict(color="#e74c3c", width=2)
            ))
            fig_pca_var.update_layout(
                title="PCA Explained Variance",
                yaxis_title="Variance Ratio",
                height=300,
                margin=dict(t=30, b=30),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_pca_var, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Cluster Visualization & Profiles")

    col_scatter, col_sizes = st.columns([3, 2])

    with col_scatter:
        st.subheader("PCA Scatter Plot")
        if pca_coords is not None:
            pca_coords["ClusterLabel"] = pca_coords["Cluster"].apply(
                lambda x: "Noise" if x == -1 else f"Cluster {x}"
            )

            n_dims = len([c for c in pca_coords.columns if c.startswith("PC")])
            if n_dims >= 2:
                fig_scatter = px.scatter(
                    pca_coords, x="PC1", y="PC2",
                    color="ClusterLabel",
                    title="Customer Segments in PCA Space",
                    opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    height=500
                )
                fig_scatter.update_layout(
                    legend_title="Segment",
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                if n_dims >= 3:
                    with st.expander("🌐 3D PCA Scatter"):
                        fig_3d = px.scatter_3d(
                            pca_coords, x="PC1", y="PC2", z="PC3",
                            color="ClusterLabel",
                            opacity=0.5,
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            height=500
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

    with col_sizes:
        st.subheader("Cluster Sizes")
        if profiles_df is not None:
            fig_sizes = px.bar(
                profiles_df, x="Cluster", y="Count",
                color="Cluster",
                text="Pct",
                title="Customers per Segment",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=400
            )
            fig_sizes.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_sizes.update_layout(showlegend=False, yaxis_title="Customer Count")
            st.plotly_chart(fig_sizes, use_container_width=True)

            # Pie chart
            fig_pie = px.pie(
                profiles_df, values="Count", names="Cluster",
                title="Segment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Cluster profiles comparison
    st.subheader("📊 Cluster Comparison")
    if profiles_df is not None:
        col_a, col_b = st.columns(2)

        with col_a:
            # CLTV by cluster
            if "Mean_CLTV" in profiles_df.columns:
                fig_cltv = px.bar(
                    profiles_df, x="Cluster", y="Mean_CLTV",
                    color="Cluster", title="Mean CLTV by Segment",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    height=400
                )
                fig_cltv.update_layout(showlegend=False, yaxis_title="Mean CLTV")
                st.plotly_chart(fig_cltv, use_container_width=True)
            elif "Mean_TotalCharges" in profiles_df.columns:
                fig_tc = px.bar(
                    profiles_df, x="Cluster", y="Mean_TotalCharges",
                    color="Cluster", title="Mean Total Charges by Segment",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    height=400
                )
                fig_tc.update_layout(showlegend=False, yaxis_title="Mean Total Charges")
                st.plotly_chart(fig_tc, use_container_width=True)

        with col_b:
            # Churn rate by cluster
            if "Churn_Rate" in profiles_df.columns:
                fig_churn = px.bar(
                    profiles_df, x="Cluster", y="Churn_Rate",
                    color="Cluster", title="Observed Churn Rate by Segment",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    height=400
                )
                fig_churn.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig_churn, use_container_width=True)
                st.caption("⚠️ Churn rates are observed post-clustering, not predictions.")

        # Monthly charges
        if "Mean_MonthlyCharges" in profiles_df.columns:
            fig_mc = px.bar(
                profiles_df, x="Cluster", y="Mean_MonthlyCharges",
                color="Cluster", title="Mean Monthly Charges by Segment",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=350
            )
            fig_mc.update_layout(showlegend=False, yaxis_title="Mean Monthly Charges ($)")
            st.plotly_chart(fig_mc, use_container_width=True)

    # Full profile table
    st.subheader("📋 Full Cluster Profile Table")
    if profiles_df is not None:
        st.dataframe(
            profiles_df.style.format({
                "Pct": "{:.1f}%",
                "Mean_Tenure": "{:.1f}",
                "Mean_MonthlyCharges": "${:.2f}",
                "Mean_TotalCharges": "${:.2f}",
                "Mean_CLTV": "${:.2f}",
                "Churn_Rate": "{:.1f}%",
            }, na_rep="N/A"),
            use_container_width=True,
            hide_index=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("HDBSCAN Hyperparameter Experiments")

    col_exp_a, col_exp_b = st.columns(2)

    with col_exp_a:
        st.subheader("Experiment A: Direct HDBSCAN")
        if grid_A:
            df_A = pd.DataFrame(grid_A)
            display_cols = ["min_cluster_size", "min_samples", "n_clusters", "noise_pct",
                           "silhouette", "davies_bouldin", "calinski_harabasz", "runtime_s"]
            display_cols = [c for c in display_cols if c in df_A.columns]
            st.dataframe(df_A[display_cols], use_container_width=True, hide_index=True)

            # Heatmap of silhouette scores
            valid_A = df_A[df_A["silhouette"].notna()].copy()
            if len(valid_A) > 1:
                pivot_A = valid_A.pivot_table(
                    values="silhouette", index="min_cluster_size",
                    columns="min_samples", aggfunc="first"
                )
                fig_heat_A = px.imshow(
                    pivot_A, text_auto=".3f",
                    title="Silhouette Score Heatmap (Exp A)",
                    color_continuous_scale="Viridis",
                    labels=dict(x="min_samples", y="min_cluster_size", color="Silhouette")
                )
                st.plotly_chart(fig_heat_A, use_container_width=True)

    with col_exp_b:
        st.subheader("Experiment B: PCA + HDBSCAN")
        if grid_B:
            df_B = pd.DataFrame(grid_B)
            display_cols = ["min_cluster_size", "min_samples", "n_clusters", "noise_pct",
                           "silhouette", "davies_bouldin", "calinski_harabasz", "runtime_s"]
            display_cols = [c for c in display_cols if c in df_B.columns]
            st.dataframe(df_B[display_cols], use_container_width=True, hide_index=True)

            valid_B = df_B[df_B["silhouette"].notna()].copy()
            if len(valid_B) > 1:
                pivot_B = valid_B.pivot_table(
                    values="silhouette", index="min_cluster_size",
                    columns="min_samples", aggfunc="first"
                )
                fig_heat_B = px.imshow(
                    pivot_B, text_auto=".3f",
                    title="Silhouette Score Heatmap (Exp B)",
                    color_continuous_scale="Viridis",
                    labels=dict(x="min_samples", y="min_cluster_size", color="Silhouette")
                )
                st.plotly_chart(fig_heat_B, use_container_width=True)

    # Churn score experiment
    if churn_score_exp:
        st.markdown("---")
        st.subheader("⚠️ Churn Score Impact (Leakage Comparison)")
        st.warning("ChurnScore is outcome-derived. Including it risks target leakage. "
                  "Shown for comparison only; primary results exclude it.")
        col_cs1, col_cs2 = st.columns(2)
        with col_cs1:
            st.markdown("**Without ChurnScore (Primary)**")
            without = churn_score_exp.get("without_churn_score", {})
            st.json(without)
        with col_cs2:
            st.markdown("**With ChurnScore (Comparison Only)**")
            with_cs = churn_score_exp.get("with_churn_score", {})
            st.json(with_cs)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: ALGORITHM COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Algorithm Comparison")

    if comparisons:
        df_comp = pd.DataFrame(comparisons)
        display_cols = ["algorithm", "n_clusters", "noise_pct", "silhouette",
                       "davies_bouldin", "calinski_harabasz", "runtime_s"]
        display_cols = [c for c in display_cols if c in df_comp.columns]

        st.dataframe(df_comp[display_cols], use_container_width=True, hide_index=True)

        st.info("**Note:** Density-based algorithms (HDBSCAN, DBSCAN) can label points as noise/outliers, "
               "whereas K-Means and Agglomerative assign every point to a cluster. "
               "Metrics are computed on non-noise points for density-based methods, making direct "
               "comparison nuanced.")

        # Visual comparison
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            valid_comp = df_comp[df_comp["silhouette"].notna()]
            if len(valid_comp) > 0:
                fig_comp_sil = px.bar(
                    valid_comp, x="algorithm", y="silhouette",
                    color="algorithm", title="Silhouette Score Comparison",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    height=400
                )
                fig_comp_sil.update_layout(showlegend=False)
                st.plotly_chart(fig_comp_sil, use_container_width=True)

        with col_m2:
            fig_comp_noise = px.bar(
                df_comp, x="algorithm", y="noise_pct",
                color="algorithm", title="Noise Percentage",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=400
            )
            fig_comp_noise.update_layout(showlegend=False, yaxis_title="Noise %")
            st.plotly_chart(fig_comp_noise, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: STABILITY
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Cluster Stability Analysis")

    if stability:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Mean ARI", f"{stability['ari_mean']:.4f}" if stability.get("ari_mean") else "N/A")
        with col_s2:
            st.metric("Std ARI", f"{stability['ari_std']:.4f}" if stability.get("ari_std") else "N/A")
        with col_s3:
            st.metric("Valid Runs", f"{stability['n_valid_runs']}/{stability['n_iterations']}")

        st.markdown(f"**Method:** {stability.get('noise_handling', 'N/A')}")
        st.markdown(f"**Subsample fraction:** {stability.get('subsample_frac', 'N/A')}")
        st.markdown(f"**HDBSCAN params:** min_cluster_size={stability.get('min_cluster_size')}, "
                   f"min_samples={stability.get('min_samples')}")

        if stability.get("ari_scores"):
            fig_ari = go.Figure()
            fig_ari.add_trace(go.Scatter(
                x=list(range(1, len(stability["ari_scores"]) + 1)),
                y=stability["ari_scores"],
                mode="lines+markers",
                marker=dict(size=8, color="#667eea"),
                line=dict(width=2, color="#667eea"),
                name="ARI"
            ))
            fig_ari.add_hline(y=stability["ari_mean"], line_dash="dash",
                            line_color="red", annotation_text=f"Mean={stability['ari_mean']:.4f}")
            fig_ari.update_layout(
                title="Adjusted Rand Index Across Subsample Iterations",
                xaxis_title="Iteration",
                yaxis_title="ARI",
                height=400,
                yaxis_range=[0, 1.05]
            )
            st.plotly_chart(fig_ari, use_container_width=True)

        # Interpretation
        ari_mean = stability.get("ari_mean", 0)
        if ari_mean is not None:
            if ari_mean > 0.8:
                st.success(f"✅ **High stability** (ARI={ari_mean:.4f}): Cluster assignments are highly "
                          "consistent across subsamples.")
            elif ari_mean > 0.5:
                st.warning(f"⚠️ **Moderate stability** (ARI={ari_mean:.4f}): Cluster structure is "
                          "partially preserved across subsamples. Some reassignment variability expected.")
            else:
                st.error(f"❌ **Low stability** (ARI={ari_mean:.4f}): Cluster assignments vary "
                        "substantially across subsamples. Results should be interpreted cautiously.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Customer Personas & Business Insights")

    if profiles_df is not None and "Persona" in profiles_df.columns:
        for _, row in profiles_df.iterrows():
            cluster_name = row["Cluster"]
            persona = row.get("Persona", cluster_name)

            if row["ClusterID"] == -1:
                icon = "🔇"
            elif "High-churn" in persona:
                icon = "🚨"
            elif "Loyal" in persona:
                icon = "💚"
            elif "High-spend" in persona:
                icon = "💎"
            elif "Low-spend" in persona:
                icon = "🔵"
            elif "New" in persona:
                icon = "🌱"
            else:
                icon = "📌"

            with st.expander(f"{icon} {cluster_name}: {persona} ({row['Count']:,} customers, {row['Pct']}%)"):
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    if "Mean_Tenure" in row and pd.notna(row["Mean_Tenure"]):
                        st.metric("Avg Tenure (months)", f"{row['Mean_Tenure']:.1f}")
                    if "Mean_MonthlyCharges" in row and pd.notna(row["Mean_MonthlyCharges"]):
                        st.metric("Avg Monthly Charges", f"${row['Mean_MonthlyCharges']:.2f}")
                with col_p2:
                    if "Mean_TotalCharges" in row and pd.notna(row["Mean_TotalCharges"]):
                        st.metric("Avg Total Charges", f"${row['Mean_TotalCharges']:.2f}")
                    if "Mean_CLTV" in row and pd.notna(row["Mean_CLTV"]):
                        st.metric("Avg CLTV", f"${row['Mean_CLTV']:.2f}")
                with col_p3:
                    if "Churn_Rate" in row and pd.notna(row["Churn_Rate"]):
                        st.metric("Churn Rate", f"{row['Churn_Rate']:.1f}%")
                    if "Mean_Referrals" in row and pd.notna(row["Mean_Referrals"]):
                        st.metric("Avg Referrals", f"{row['Mean_Referrals']:.1f}")

                # Show recommendation
                if recommendations:
                    for rec in recommendations:
                        if rec.get("Cluster") == cluster_name:
                            st.markdown(f"**💡 Recommendation:** {rec['Recommendation']}")
                            break

                # Extra details
                details = {}
                if "Top_Contract" in row and pd.notna(row["Top_Contract"]):
                    details["Most Common Contract"] = row["Top_Contract"]
                if "Top_InternetService" in row and pd.notna(row["Top_InternetService"]):
                    details["Most Common Internet"] = row["Top_InternetService"]
                if "Top_PaymentMethod" in row and pd.notna(row["Top_PaymentMethod"]):
                    details["Most Common Payment"] = row["Top_PaymentMethod"]
                if details:
                    st.markdown("**Additional Profile:**")
                    for k, v in details.items():
                        st.markdown(f"- {k}: {v}")

    st.markdown("---")
    st.subheader("⚠️ Limitations & Notes")
    st.markdown("""
    - **Unsupervised clustering does not predict churn.** Observed churn rates are post-hoc descriptive statistics, not forward-looking predictions.
    - **HDBSCAN noise points** are customers that don't fit neatly into any cluster. They may represent edge cases or unique profiles.
    - **Cluster names** are automatically derived from observed characteristics and should be validated with domain expertise.
    - **Stability** depends on dataset size, feature selection, and HDBSCAN parameters. Low stability suggests the cluster structure may be fragile.
    - **Business recommendations** are suggestions based on observed patterns and should be validated before implementation.
    """)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9rem;'>"
    "Customer Segmentation Research Project | HDBSCAN-Based Persona Discovery | 2026"
    "</div>",
    unsafe_allow_html=True
)
