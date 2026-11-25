
import pandas as pd


def load_and_clean_data(file_path="data/student_performance.csv"):
    try:
        df = pd.read_csv(file_path)

        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        df = df.drop_duplicates()
        df = df.dropna(how="all")

        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("Unknown")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. "
            "Please check your data folder."
        )


def apply_filters(df, column=None, value=None):
    if column is None or value is None:
        return df

    if column not in df.columns:
        return df

    return df[df[column].astype(str) == str(value)]


def make_recommendation(row):
    recommendations = []

    for column in row.index:
        value = row[column]

        if isinstance(value, (int, float)):
            if value < 75:
                recommendations.append(
                    f"Needs improvement in {column}."
                )
            elif value >= 90:
                recommendations.append(
                    f"Excellent performance in {column}."
                )

    if not recommendations:
        return "Continue maintaining good academic performance."

    return " ".join(recommendations)
