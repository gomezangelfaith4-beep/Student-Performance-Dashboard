import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = ["year_level","prior_attendance_rate","prior_math_grade","prior_english_grade",
            "prior_science_grade","prior_general_average","prior_failed_subjects"]
TARGET = "risk_level"

def train_risk_models(df):
    data = df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()
    if data[TARGET].nunique() < 2:
        return {"error":"At least two risk categories are needed to train a classifier."}
    X, y = data[FEATURES], data[TARGET]
    counts = y.value_counts()
    if counts.min() < 2:
        return {"error":"Each risk category needs at least two records for a stratified train/test split."}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    numeric_pipe = Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    prep = ColumnTransformer([("numeric",numeric_pipe,FEATURES)])
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    }
    out = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess",prep),("model",estimator)])
        pipe.fit(X_train,y_train)
        pred = pipe.predict(X_test)
        labels = sorted(y.unique())
        out[name] = {
            "accuracy":accuracy_score(y_test,pred),
            "precision":precision_score(y_test,pred,average="weighted",zero_division=0),
            "recall":recall_score(y_test,pred,average="weighted",zero_division=0),
            "f1":f1_score(y_test,pred,average="weighted",zero_division=0),
            "confusion_matrix":confusion_matrix(y_test,pred,labels=labels),
            "labels":labels,
            "pipeline":pipe
        }
    importances = out["Random Forest"]["pipeline"].named_steps["model"].feature_importances_
    out["Random Forest"]["importance"] = pd.DataFrame({"feature":FEATURES,"importance":importances})
    return out

def run_student_clustering(df, k=3):
    features = ["prior_attendance_rate","prior_general_average","prior_failed_subjects"]
    if len(df) <= k:
        return {"error":f"Select at least {k+1} student records to create {k} clusters."}
    x = df[features].copy()
    x = x.fillna(x.median())
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(xs)
    if len(set(labels)) < 2:
        return {"error":"K-Means created fewer than two distinct clusters. Try a different K."}
    result = df.copy()
    result["cluster"] = labels.astype(str)
    score = silhouette_score(xs, labels) if len(set(labels)) > 1 and len(df) > k else float("nan")
    return {"data":result,"silhouette":score,"model":model}
