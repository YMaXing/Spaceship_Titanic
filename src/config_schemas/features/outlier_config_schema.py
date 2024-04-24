from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class outlier_Config:
    local_data_dir = "data/EDA_export"
    local_save_dir = "data/imputed"
    expenses = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    label = "Transported"
    max_iter = 5


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="outlier_config_schema", node=outlier_Config)
