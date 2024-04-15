from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass


@dataclass
class Process_data_Config:
    dvc_remote_name: str = "gcs-storage"
    dvc_remote_url: str = "gs://spaceship_titanic/raw_data"
    dvc_raw_data_folder: str = "data/raw"
    gcp_project_id: str = "555402469041"
    gcp_secret_id: str = "SpaceshipTitanic_token"
    gcp_version_id: str = "1"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="process_data_schema", node=Process_data_Config, group="data")
