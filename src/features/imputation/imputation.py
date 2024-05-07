import pandas as pd
import numpy as np
import logging

from src.utils.imputation_utils import read_data, save_data
from src.utils.config_utils import get_config
from src.config_schemas.features.imputation_config_schema import imputation_Config
from src.features.imputation.CatBoost_imputer import iter_cv_catboost_imputer


@get_config(config_path="../configs/features", config_name="imputation_config")
def imputation(config: imputation_Config) -> iter_cv_catboost_imputer:
    # Read training and test data from local directory
    try:
        df_train, df_test = read_data(config.local_data_dir)
    except Exception as e:
        logging.error(f"Failed to read data: {e}")
        return None

    expenses = config.expenses
    # First, impute both training and test data based on EDA observations
    df_train = EDA_imputer(df_train, expenses)
    df_test = EDA_imputer(df_test, expenses)
    logging.info("Imputation based on EDA observations is done.")
    # Then, impute the remaining missing values with customized CatBoost imputer
    Catboost_imputer = iter_cv_catboost_imputer(
        label=config.label,
        max_iter=config.max_iter,
    )
    df_train = Catboost_imputer.fit_transform(df_train)
    df_test = Catboost_imputer.transform(df_test)
    # Clip the negative imputed expenses value 
    df_train[expenses] = df_train[expenses].clip(lower=0)
    df_test[expenses] = df_test[expenses].clip(lower=0)

    save_data(df_train, df_test, config.local_save_dir)

    Catboost_imputer.plot_training_error()


def EDA_imputer(df: pd.DataFrame, expenses: list) -> pd.DataFrame:
    """
    We impute based on the observations we had in the EDA notebook
    """
    # Impute missing expense with 0 for those who were cryosleeping
    df.loc[df["CryoSleep"] == True, expenses] = df.loc[df["CryoSleep"] == True, expenses].fillna(0)

    # Impute missing VIP with False since for those from Earth and those no older than 18
    df.loc[(df["HomePlanet"] == "Earth") | (df["Age"] <= 18), "VIP"] = False

    # Impute missing expenses with 0 for anyone no older than 12
    df.loc[df["Age"] <= 12, expenses] = df.loc[df["Age"] <= 12, expenses].fillna(0)

    # Passengers living on deck A, B and C are 100% from Europa
    df.loc[df["Cabin_deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = df.loc[df["Cabin_deck"].isin(["A", "B", "C", "T"]), "HomePlanet"].fillna("Europa")

    # Passengers living on deck G are 100% from Earth
    df.loc[df["Cabin_deck"] == "G", "HomePlanet"] = df.loc[df["Cabin_deck"] == "G", "HomePlanet"].fillna("Earth")

    # Impute missing HomePlanet and Cabin_side within each group since they take unique value in every group
    # Fill in the missing values in "HomePlanet" and "cabin_side" for any groups with at least 2 passengers
    # with the same value as that of the other group members if not all of theirs are missing
    df = df.merge(df.groupby("ID_group")["HomePlanet"].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index(), on='ID_group')
    df = df.merge(df.groupby("ID_group")["Cabin_side"].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index(), on='ID_group')
    df.drop(columns=['HomePlanet_x', 'Cabin_side_x'], inplace=True)
    df.rename(columns={'HomePlanet_y': 'HomePlanet', 'Cabin_side_y': 'Cabin_side'}, inplace=True)

    # Impute missing HomePlanet with Europa for who spent over a total of 2500 in FoodCourt, VRDeck and Spa
    # Step 1: Calculate the condition and get indices
    condition = (df["FoodCourt"] + df["VRDeck"] + df["Spa"] > 2500)
    indices = df[condition].index
    # Step 2: Use loc to modify "HomePlanet" safely
    df.loc[indices, "HomePlanet"] = df.loc[indices, "HomePlanet"].fillna("Europa")

    # Drop ID_group
    df = df.drop(columns="ID_group")

    return df


if __name__ == "__main__":
    imputation()  # type: ignore
