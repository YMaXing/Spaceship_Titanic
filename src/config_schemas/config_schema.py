from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass


@dataclass
class Config:
    dvc_remote_name: str = "gcs-storage"
    dvc_remote_url: str = "gs://spaceship_titanic/raw_data"
    dvc_raw_data_folder: str = "data/raw"

    local_test_data_dir: str = "data/feature-selected"
    local_sample_submission_dir: str = "data/raw/sample_submission.csv"
    final_data_folder: str = "data/final"
    final_model_name: str = "final_model"
    final_encoding: str = "final_encoding"
    final_model_run_id: str = "final_model_run_id"
    label: str = "Transported"
    majority_threshold: float = 5


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
