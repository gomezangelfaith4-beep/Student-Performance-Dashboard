
import pandas as pd


def load_and_clean_data(file_path):

    # Read CSV dataset
    df = pd.read_csv(file_path)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove records with missing required values
    required_columns = [
        "Student_ID",
        "Student_Name",
        "Gender",
        "Attendance",
        "Study_Hours",
        "Assignment_Score",
        "Exam_Score"
    ]

    df = df.dropna(subset=required_columns)

    # Convert numerical columns
    numeric_columns = [
        "Attendance",
        "Study_Hours",
        "Assignment_Score",
        "Exam_Score"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=numeric_columns)

    # Calculate final grade
    df["Final_Grade"] = (
        df["Assignment_Score"] * 0.40
        + df["Exam_Score"] * 0.60
    ).round(2)

    # Classify student performance
    def classify_performance(grade):

        if grade >= 90:
            return "Excellent"

        elif grade >= 75:
            return "Good"

        else:
            return "Needs Improvement"

    df["Performance"] = df["Final_Grade"].apply(
        classify_performance
    )

    return df


def apply_filters(
    df,
    gender="All",
    performance="All"
):

    filtered_df = df.copy()

    if gender != "All":
        filtered_df = filtered_df[
            filtered_df["Gender"] == gender
        ]

    if performance != "All":
        filtered_df = filtered_df[
            filtered_df["Performance"] == performance
        ]

    return filtered_df


def make_recommendation(row):

    grade = row["Final_Grade"]
    attendance = row["Attendance"]

    if grade < 75 or attendance < 75:
        return "Needs Improvement"

    elif grade >= 90 and attendance >= 90:
        return "Excellent Performance"

    else:
        return "Good Performance"
