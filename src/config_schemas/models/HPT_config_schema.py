from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class HPT_Config:
    local_data_dir: str = "data/feature-selected"
    local_save_dir: str = "data/models"
    label: str = "Transported"
    model_name: str = "RandomForest"
    model_params: dict = field(default_factory=lambda: {"n_estimators": 100, "max_depth": 2, "random_state": 42})
    cv: int = 5
    scoring: str = "roc_auc"
    n_iter: int = 10
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 1


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="HPT_config_schema", node=HPT_Config)
