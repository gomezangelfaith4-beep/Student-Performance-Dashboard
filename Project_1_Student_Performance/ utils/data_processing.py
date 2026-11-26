import pandas as pd


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    return df


def apply_filters(df, column=None, value=None):
    if column and value is not None:
        df = df[df[column] == value]
    return df


def make_recommendation(row):
    return "Needs Improvement"
