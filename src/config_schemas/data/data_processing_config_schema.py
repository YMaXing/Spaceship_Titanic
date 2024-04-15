from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from src.config_schemas.infrastructure.gcp_config_schema import GCP_Config
from omegaconf import MISSING


@dataclass
class data_processing_Config:
    data_local_save_dir = "data/raw"
    github_user_name: str = "YMaXing"
    version: str = MISSING
    dvc_remote_repo: str = "github.com/YMaXing/Spaceship_Titanic.git"
    dvc_data_folder: str = "data/raw"

    GCP_Config = GCP_Config()
    gcp_project_id: str = GCP_Config.gcp_project_id
    gcp_secret_id: str = GCP_Config.gcp_secret_id


def setup_config() -> None:

    cs = ConfigStore.instance()
    cs.store(name="data_processing_schema", node=data_processing_Config)
