from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class feature_selection_Config:
    local_save_dir = "data/feature-selected/NE/No_Cryo"
    local_encoded_dir = "data/encoded/double"
    local_unencoded_dir = "data/engineered"

    label: list[str] = field(default_factory=lambda: ["Transported"])

    unencoded_features: list = field(
        default_factory=lambda: [
            "Destination",
            "Group_size",
            "HomePlanet",
            "Age_group",
            "Consumption_High_End",
            "Consumption_Basic"
        ]
    )

    encoded: list = field(default_factory=lambda: ["Mixed"])

    Mixed_features: list = field(
        default_factory=lambda: [
            "Destination",
            "CryoSleep_False",
            "RoomService",
            "Cabin_deck",
            "Consumption_High_End",
            "Consumption_Basic"
        ]
    )


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="feature_selection_config_schema", node=feature_selection_Config)
