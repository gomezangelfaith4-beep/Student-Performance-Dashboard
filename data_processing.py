from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = [
    "student_code","school_year","year_level","section","prior_attendance_rate",
    "prior_math_grade","prior_english_grade","prior_science_grade","prior_general_average",
    "prior_failed_subjects","attendance_rate","math_grade","english_grade","science_grade",
    "general_average","failed_subjects","academic_status","risk_level","intervention_records"
]

def load_and_clean_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put student_performance.csv inside the data folder.")
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_cols))
    before = len(df)
    df = df.drop_duplicates(subset=["student_code"]).copy()
    duplicates = before - len(df)
    numeric = [c for c in df.columns if c.endswith("_grade") or c.endswith("_rate") or c in ["general_average","prior_general_average","failed_subjects","prior_failed_subjects","year_level"]]
    missing_count = int(df[numeric].isna().sum().sum())
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in ["section","school_year","academic_status","risk_level","intervention_records","sex"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)
    df["student_code"] = df["student_code"].astype(str)
    df["year_level"] = df["year_level"].astype(int)
    df["failed_subjects"] = df["failed_subjects"].clip(lower=0).astype(int)
    df["prior_failed_subjects"] = df["prior_failed_subjects"].clip(lower=0).astype(int)
    for c in [x for x in df.columns if x.endswith("_grade") or x in ["general_average","prior_general_average"]]:
        df[c] = df[c].clip(0,100)
    for c in ["attendance_rate","prior_attendance_rate"]:
        df[c] = df[c].clip(0,100)
    return df, {"missing_values_filled": missing_count, "duplicate_rows_removed": duplicates}

def apply_filters(df, years, levels, sections, risks):
    mask = df["school_year"].isin(years) & df["year_level"].isin(levels) & df["section"].isin(sections) & df["risk_level"].isin(risks)
    return df.loc[mask].copy()

def make_recommendation(row):
    if row["risk_level"] == "High" or row["general_average"] < 70 or row["failed_subjects"] >= 2 or row["attendance_rate"] < 70:
        if row["attendance_rate"] < 70:
            return "Attendance follow-up + adviser meeting"
        if row["failed_subjects"] >= 2:
            return "Remedial class + subject tutorial"
        return "Guidance referral + parent/guardian conference"
    if row["risk_level"] == "Moderate" or row["general_average"] < 80 or row["attendance_rate"] < 85:
        return "Academic advising + progress monitoring"
    return "Continue regular monitoring / enrichment"
