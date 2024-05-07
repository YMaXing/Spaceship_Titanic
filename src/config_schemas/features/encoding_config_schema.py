from hydra.core.config_store import ConfigStore
from matplotlib.pyplot import hot
from pydantic.dataclasses import dataclass
from dataclasses import field
from omegaconf import MISSING


@dataclass
class encoding_Config:
    local_data_dir = "data/engineered"
    local_save_dir = "data/encoded/double"
    encoder_name: str = "WOEEncoder"
    cat_validation: str = "Double"
    label: str = "Transported"

    # For mixed encoder
    supported_encoders: list = field(
        default_factory=lambda: ["WOEEncoder", "CatBoostEncoder", "MEstimateEncoder", "OneHotEncoder"]
    )
    OneHotEncoder_cols: list = field(default_factory=lambda: ["CryoSleep"])
    WOEEncoder_cols: list = field(default_factory=lambda: ["Destination", "Cabin_deck", "HomePlanet"])
    CatBoostEncoder_cols: list = field(default_factory=list)
    MEstimateEncoder_cols: list = field(default_factory=list)


def setup_config() -> None:

    cs = ConfigStore.instance()
    cs.store(name="encoding_config_schema", node=encoding_Config)
