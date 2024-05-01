from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class feature_selection_Config:
    local_encoded_dir = "data/encoded"
    local_unencoded_dir = "data/engineered"
    local_save_dir = "data/feature-selected"

    label: list[str] = field(default_factory=lambda: ["Transported"])

    unencoded_features: list = field(
        default_factory=lambda: [
            "CryoSleep",
            "Destination",
            "VIP",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
            "Cabin_deck",
            "Group_size",
            "HomePlanet",
            "Consumption_High_End",
            "Consumption_Basic",
            "Consumption_Total",
            "Age_group",
        ]
    )

    encoded: list = field(default_factory=lambda: ["MEstimate", "Mixed"])

    MEstimate_features: list = field(
        default_factory=lambda: [
            "CryoSleep",
            "Destination",
            "RoomService",
            "FoodCourt",
            "Spa",
            "Cabin_deck",
            "HomePlanet",
            "Consumption_High_End",
            "Consumption_Basic",
            "Consumption_Total",
            "Age_group",
        ]
    )

    Mixed_features: list = field(
        default_factory=lambda: [
            "Destination",
            "CryoSleep_False",
            "VRDeck",
            "Spa",
            "Cabin_deck",
            "HomePlanet",
            "Consumption_High_End",
            "Consumption_Basic",
            "Consumption_Total",
        ]
    )


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="feature_selection_config_schema", node=feature_selection_Config)
