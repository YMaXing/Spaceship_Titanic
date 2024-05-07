import pandas as pd
from src.utils.config_utils import get_config
from src.config_schemas.features.feature_engineering_config_schema import feature_engineering_Config
import logging


@get_config(config_path="../configs/features", config_name="feature_engineering_config")
def feature_engineering(config: feature_engineering_Config) -> None:
    # Read imputed training and test data from local directory
    df_train = pd.read_csv(config.local_data_dir + "/train.csv")
    df_test = pd.read_csv(config.local_data_dir + "/test.csv")
    logging.info("Data loaded successfully")

    # Create "Consumption_High_End" feature
    df_train["Consumption_High_End"] = df_train["RoomService"] + df_train["Spa"] + df_train["VRDeck"]
    df_test["Consumption_High_End"] = df_test["RoomService"] + df_test["Spa"] + df_test["VRDeck"]

    # Create "Consumption_Basic" feature
    df_train["Consumption_Basic"] = df_train["FoodCourt"] + df_train["ShoppingMall"]
    df_test["Consumption_Basic"] = df_test["FoodCourt"] + df_test["ShoppingMall"]

    # Bin Age into age groups
    # Define the bins and labels
    bins = [0, 18, 40, 60, float("inf")]  # Using float('inf') to include all higher numbers
    labels = ["Minor", "Young adults", "Middle-aged", "Senior"]
    df_train["Age_group"] = pd.cut(
        df_train["Age"], bins=bins, labels=labels, right=False
    )  # right=False means the intervals are left-inclusive
    df_test["Age_group"] = pd.cut(
        df_test["Age"], bins=bins, labels=labels, right=False
    )

    # Remove corresponding old features
    df_train = df_train.drop(
        columns=["Age", "ID_num", "Cabin_side"]
    )
    df_test = df_test.drop(
        columns=["Age", "ID_num", "Cabin_side"]
    )

    # Save engineered datasets
    df_train.to_csv(config.local_save_dir + "/train.csv", index=False)
    df_test.to_csv(config.local_save_dir + "/test.csv", index=False)
    logging.info("Data saved successfully")


if __name__ == "__main__":
    feature_engineering()  # type: ignore
