import pandas as pd


def read_data(local_data_dir: str) -> pd.DataFrame:
    df_train = pd.read_csv(local_data_dir + "/train.csv")
    df_test = pd.read_csv(local_data_dir + "/test.csv")
    return df_train, df_test


def save_data(train: pd.DataFrame, test: pd.DataFrame, local_save_dir: str) -> None:
    train.to_csv(local_save_dir + "/train.csv", index=False)
    test.to_csv(local_save_dir + "/test.csv", index=False)


def missing_index(df, features):
    missing_row_index = {}

    for feature in features:
        missing_row_index[feature] = df[df[feature].isna()].index

    return missing_row_index


def get_features(df: pd.DataFrame = None, missing_type: str = "cat", label: str = None):

    if missing_type == "cat":
        features = [
            feature
            for feature in df.columns
            if df[feature].dtype == "O" and df[feature].isna().sum() > 0 and feature != label
        ]
    elif missing_type == "num":
        features = [
            feature
            for feature in df.columns
            if df[feature].dtype != "O" and df[feature].isna().sum() > 0 and feature != label
        ]
    else:
        raise TypeError(f"The missing_type you gave is {missing_type}, but it can only be 'cat' or 'num'!")

    return features


def fill_placeholder(df: pd.DataFrame = None, features: list = None):

    for feature in features:
        if df[feature].dtype == "O":
            df[feature] = df[feature].fillna(f"Missing_{feature}")
        elif df[feature].dtype != "O":
            df[feature] = df[feature].fillna(df[feature].median())

    return df
