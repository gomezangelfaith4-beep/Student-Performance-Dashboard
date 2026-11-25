import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def _prior_features(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column.startswith("prior_") and df[column].notna().any()
    ]


def train_risk_models(df: pd.DataFrame) -> dict[str, object]:
    """Train and evaluate tree models using prior-period data only."""
    target_column = "risk_level"
    features = _prior_features(df)
    if df.empty:
        return {"error": "No student records are available for model training."}
    if target_column not in df.columns or not features:
        return {
            "error": (
                "Modeling requires a risk_level target and numeric prior-period "
                "academic fields."
            )
        }

    model_data = df[features + [target_column]].dropna(subset=[target_column])
    labels = model_data[target_column].astype(str)
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        return {"error": "At least two risk categories are needed to train models."}
    if class_counts.min() < 2:
        return {
            "error": (
                "Each risk category needs at least two records for a stratified "
                "train/test evaluation."
            )
        }

    x_train, x_test, y_train, y_test = train_test_split(
        model_data[features],
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    labels_order = sorted(labels.unique())
    estimators = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results: dict[str, object] = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=labels_order
            ).tolist(),
            "labels": labels_order,
        }
        if name == "Random Forest":
            results[name]["importance"] = (
                pd.DataFrame(
                    {
                        "feature": features,
                        "importance": estimator.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

    results["test_records"] = len(y_test)
    results["feature_names"] = features
    return results


def run_student_clustering(
    df: pd.DataFrame, k: int = 3
) -> dict[str, object]:
    """Cluster students using available numeric prior-period performance data."""
    features = _prior_features(df)
    if len(df) < 2:
        return {"error": "At least two student records are needed for clustering."}
    if not features:
        return {
            "error": "Clustering requires numeric prior-period academic fields."
        }
    if not isinstance(k, int) or k < 2 or k >= len(df):
        return {
            "error": "Choose a cluster count of at least 2 and less than the record count."
        }

    numeric = df[features].replace([np.inf, -np.inf], np.nan)
    if numeric.nunique(dropna=False).max() < 2:
        return {"error": "The available prior-period data has no variation to cluster."}

    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaled = preprocessing.fit_transform(numeric)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    distinct_clusters = np.unique(labels)
    if len(distinct_clusters) < 2:
        return {
            "error": "The data formed fewer than two distinct clusters; try a different K."
        }

    result = df.copy()
    result["cluster"] = labels
    score = (
        silhouette_score(scaled, labels)
        if len(distinct_clusters) < len(df)
        else 0.0
    )
    return {
        "data": result,
        "silhouette": float(score),
        "cluster_count": int(len(distinct_clusters)),
        "features": features,
    }
