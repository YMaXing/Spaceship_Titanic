import logging
import subprocess
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"[socket.gethostname()]:{name}")


def run_shell_command(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout


def read_data(local_data_dir: str) -> pd.DataFrame:
    df_train = pd.read_csv(local_data_dir + "/train.csv")
    df_test = pd.read_csv(local_data_dir + "/test.csv")
    return df_train, df_test


def save_data(train: pd.DataFrame, test: pd.DataFrame, local_save_dir: str) -> None:
    train.to_csv(local_save_dir + "/train.csv", index=False)
    test.to_csv(local_save_dir + "/test.csv", index=False)
