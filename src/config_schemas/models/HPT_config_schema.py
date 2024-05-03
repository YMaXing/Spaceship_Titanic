from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from dataclasses import field

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


@dataclass
class HPT_Config:
    local_data_dir: str = "data/feature-selected"
    local_save_dir: str = "data/models"
    label: str = "Transported"

    # Use Type for specifying the type of the classifier
    models = [(xgb.XGBClassifier, "XGBoost"),
              (CatBoostClassifier, "CatBoost"),
              (ExtraTreesClassifier, "ExtraTrees"),
              (HistGradientBoostingClassifier, "Hist"),
              (lgb.LGBMClassifier, "LGBM")]
    # Encoding types: "NE" for No Encoding, "ME" for M-Estimate Encoding, "Mixed" for Mixed One-Hot and CatBoost Encoding
    encodings: list[str] = field(default_factory=lambda: ["NE", "ME", "Mixed"])

    # Base model parameters
    base_params_dict: dict = field(
        default_factory=lambda: {
            "CatBoost": {"verbose": False, "eval_metric": "Accuracy"},
            "ExtraTrees": {"n_jobs": -1, "random_state": 42},
            "Hist": {"random_state": 42},
            "XGBoost": {
                "random_state": 42,
                "enable_categorical": True,
                "eval_metric": "error",
                "n_jobs": -1,
            },
            "LGBM": {"random_state": 42, "objective": "binary", "n_jobs": -1},
        }
    )

    directory_base = "model_artifacts/base_model"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="HPT_config_schema", node=HPT_Config)
