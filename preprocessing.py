"""
preprocessing.py - Data loading, cleaning, feature engineering for Telco Customer Churn dataset.
Supports both IBM Telco Customer Churn dataset variants.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column name mapping: normalize common variations to canonical names
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # Tenure
    "tenure months": "Tenure",
    "tenure": "Tenure",
    # Monthly charges
    "monthly charges": "MonthlyCharges",
    "monthly charge": "MonthlyCharges",
    "monthlycharges": "MonthlyCharges",
    # Total charges
    "total charges": "TotalCharges",
    "totalcharges": "TotalCharges",
    # CLTV
    "cltv": "CLTV",
    "customer lifetime value": "CLTV",
    # Churn
    "churn label": "ChurnLabel",
    "churn": "ChurnLabel",
    "churnlabel": "ChurnLabel",
    "churn value": "ChurnValue",
    "churnvalue": "ChurnValue",
    # Churn Score
    "churn score": "ChurnScore",
    "churnscore": "ChurnScore",
    # Referrals
    "number of referrals": "NumReferrals",
    "num referrals": "NumReferrals",
    # Customer ID
    "customerid": "CustomerID",
    "customer id": "CustomerID",
    # Contract
    "contract": "Contract",
    # Gender
    "gender": "Gender",
    # Senior citizen
    "senior citizen": "SeniorCitizen",
    "seniorcitizen": "SeniorCitizen",
    # Partner / Dependents
    "partner": "Partner",
    "dependents": "Dependents",
    # Phone service
    "phone service": "PhoneService",
    "phoneservice": "PhoneService",
    # Multiple lines
    "multiple lines": "MultipleLines",
    "multiplelines": "MultipleLines",
    # Internet service
    "internet service": "InternetService",
    "internetservice": "InternetService",
    "internet type": "InternetService",
    # Online security
    "online security": "OnlineSecurity",
    "onlinesecurity": "OnlineSecurity",
    # Online backup
    "online backup": "OnlineBackup",
    "onlinebackup": "OnlineBackup",
    # Device protection
    "device protection": "DeviceProtection",
    "deviceprotection": "DeviceProtection",
    # Tech support
    "tech support": "TechSupport",
    "techsupport": "TechSupport",
    # Streaming TV
    "streaming tv": "StreamingTV",
    "streamingtv": "StreamingTV",
    # Streaming movies
    "streaming movies": "StreamingMovies",
    "streamingmovies": "StreamingMovies",
    # Paperless billing
    "paperless billing": "PaperlessBilling",
    "paperlessbilling": "PaperlessBilling",
    # Payment method
    "payment method": "PaymentMethod",
    "paymentmethod": "PaymentMethod",
}

# Columns that are target/outcome derived — never use as clustering features
LEAKAGE_COLS = {"ChurnLabel", "ChurnValue", "ChurnScore", "CLTV"}

# ID columns to drop from features
ID_COLS = {"CustomerID"}

# Categorical columns suitable for one-hot encoding
CATEGORICAL_COLS = [
    "Gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def find_dataset(project_dir: str = ".") -> str | None:
    """Search for a Telco dataset CSV in common locations."""
    search_paths = [
        Path(project_dir) / "data",
        Path(project_dir),
        Path.home() / "Downloads",
        Path.home() / "Desktop",
    ]
    patterns = ["*telco*churn*.csv", "*Telco*Churn*.csv", "*WA_Fn*churn*.csv"]
    for sp in search_paths:
        if sp.exists():
            for pat in patterns:
                for f in sp.glob(pat):
                    return str(f)
    # Also check for any CSV in data/
    data_dir = Path(project_dir) / "data"
    if data_dir.exists():
        csvs = list(data_dir.glob("*.csv"))
        if len(csvs) == 1:
            return str(csvs[0])
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to canonical names."""
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = df.rename(columns=rename)
    return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load dataset, normalize columns, clean types."""
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # Convert TotalCharges to numeric (often stored as string with spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert MonthlyCharges
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Convert Tenure
    if "Tenure" in df.columns:
        df["Tenure"] = pd.to_numeric(df["Tenure"], errors="coerce")

    # Convert CLTV
    if "CLTV" in df.columns:
        df["CLTV"] = pd.to_numeric(df["CLTV"], errors="coerce")

    # Convert ChurnScore
    if "ChurnScore" in df.columns:
        df["ChurnScore"] = pd.to_numeric(df["ChurnScore"], errors="coerce")

    # Convert NumReferrals
    if "NumReferrals" in df.columns:
        df["NumReferrals"] = pd.to_numeric(df["NumReferrals"], errors="coerce")

    # Convert SeniorCitizen to numeric if string
    if "SeniorCitizen" in df.columns:
        if df["SeniorCitizen"].dtype == object:
            df["SeniorCitizen"] = df["SeniorCitizen"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)

    # Encode ChurnLabel as binary for analysis
        # Encode ChurnLabel as binary for analysis
    if "ChurnLabel" in df.columns:
        churn = df["ChurnLabel"]

        # Handle Yes/No, True/False, and numeric labels
        normalized = churn.astype("string").str.strip().str.lower()

        mapped = normalized.map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })

        numeric = pd.to_numeric(normalized, errors="coerce")

        df["ChurnBinary"] = mapped.fillna(numeric)

    elif "ChurnValue" in df.columns:
        df["ChurnBinary"] = pd.to_numeric(
            df["ChurnValue"], errors="coerce"
        )
    return df

def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return dataset metadata."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    return info


def engineer_features(df: pd.DataFrame, include_churn_score: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature matrix for clustering.

    Returns (X_df, feature_names):
        X_df: DataFrame of engineered features (not yet scaled)
        feature_names: list of column names
    """
    feat = pd.DataFrame(index=df.index)

    # --- Numeric features ---
    if "Tenure" in df.columns:
        feat["Tenure"] = df["Tenure"]

    if "MonthlyCharges" in df.columns:
        feat["MonthlyCharges"] = df["MonthlyCharges"]

    if "TotalCharges" in df.columns:
        feat["TotalCharges"] = df["TotalCharges"]

    if "NumReferrals" in df.columns:
        feat["NumReferrals"] = df["NumReferrals"]

    # Average monthly revenue (only when meaningful: Tenure > 0)
    if "TotalCharges" in feat.columns and "Tenure" in feat.columns:
        safe_tenure = df["Tenure"].replace(0, np.nan)
        feat["AvgMonthlyRevenue"] = df["TotalCharges"] / safe_tenure

    # Churn Score as optional feature (potential leakage — used only in experiment B)
    if include_churn_score and "ChurnScore" in df.columns:
        feat["ChurnScore"] = df["ChurnScore"]

    # --- Categorical features (one-hot) ---
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_present:
        dummies = pd.get_dummies(df[cat_present], drop_first=True, dtype=int)
        feat = pd.concat([feat, dummies], axis=1)

    # Drop any remaining object columns
    feat = feat.select_dtypes(include=[np.number])

    # Handle missing values — fill with median
    feat = feat.fillna(feat.median())

    # Drop columns with zero variance
    feat = feat.loc[:, feat.std() > 0]

    feature_names = list(feat.columns)
    return feat, feature_names


def scale_features(X_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return X_scaled, scaler


def prepare_data(csv_path: str, include_churn_score: bool = False):
    """
    Full preprocessing pipeline.
    Returns: df_raw, X_scaled, feature_names, scaler, dataset_info
    """
    df = load_and_clean(csv_path)
    dataset_info = get_dataset_info(df)
    X_df, feature_names = engineer_features(df, include_churn_score=include_churn_score)
    X_scaled, scaler = scale_features(X_df)

    print(f"[preprocessing] Loaded {df.shape[0]} customers, {df.shape[1]} columns")
    print(f"[preprocessing] Feature matrix: {X_scaled.shape[0]} samples × {len(feature_names)} features")
    print(f"[preprocessing] Numeric features: {[f for f in feature_names if not any(c in f for c in ['_'])][:10]}...")
    print(f"[preprocessing] Missing values filled with median. Zero-variance columns removed.")

    return df, X_scaled, feature_names, scaler, dataset_info
