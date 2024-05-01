import pandas as pd
from pyod.models.iforest import IForest
from src.utils.config_utils import get_config
from src.config_schemas.features.outlier_config_schema import outlier_Config
from src.utils.utils import read_data, save_data
import logging


@get_config(config_path="../configs/features", config_name="outlier_config")
def outlier(config: outlier_Config) -> None:
    df_train, df_test = read_data(config.local_data_dir)
    logging.info("Loading data completed")
    df_train = iso_forest(
        df_train, config.out_features, config.n_estimators, config.n_jobs, config.random_state, config.threshold
    )
    save_data(df_train, df_test, config.local_save_dir)
    logging.info("Saving data completed")


def iso_forest(df: pd.DataFrame, out_features: list[str], n_estimators: int, n_jobs: int, random_state: int, threshold: float) -> pd.DataFrame:
    iso = IForest(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state).fit(
        df[out_features]
    )
    out_probs = iso.predict_proba(df[out_features])
    df["prob_iso"] = out_probs[:, 1]
    logging.info(
        f"There are {df[df['prob_iso'] > threshold].shape[0]} or {df[df['prob_iso'] > threshold].shape[0]/df.shape[0] * 100:.2f}% outliers removed"
    )
    df = df[df["prob_iso"] <= threshold]
    df = df.drop(columns=["prob_iso"])
    return df


if __name__ == "__main__":
    outlier()  # type: ignore
