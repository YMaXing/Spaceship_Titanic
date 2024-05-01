from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class outlier_Config:
    local_data_dir = "data/imputed"
    local_save_dir = "data/outlier_removed"
    out_features = ["Age", "RoomService", "Spa", "ShoppingMall", "VRDeck", "FoodCourt"]
    n_jobs = -1
    n_estimators = 500
    random_state = 42
    threshold = 0.8

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="outlier_config_schema", node=outlier_Config)
