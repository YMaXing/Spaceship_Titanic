from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from optuna.pruners import SuccessiveHalvingPruner, HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler


@dataclass
class HPT_Config:
    local_data_dir: str = "data/feature-selected/double"
    label: str = "Transported"
    random_state: int = 42

    # Use Type for specifying the type of the classifier
    models = [(xgb.XGBClassifier, "XGBoost"),
              (CatBoostClassifier, "CatBoost"),
              (ExtraTreesClassifier, "ExtraTrees"),
              (HistGradientBoostingClassifier, "Hist"),
              (lgb.LGBMClassifier, "LGBM")]
    # Encoding types: "NE" for No Encoding, "ME" for M-Estimate Encoding, "Mixed" for Mixed One-Hot and CatBoost Encoding
    encodings: list[str] = field(default_factory=lambda: ["NE", "Mixed"])

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
    # Hyperparameter tuning parameters
    HPT_model_name: str = "DefaultModel"
    HPT_encoding: str = "DefaultEncoding"
    pruner = SuccessiveHalvingPruner
    pruner_kwargs = {"reduction_factor": 2}
    n_rungs = 4
    sampler = TPESampler
    sampler_kwargs = {}
    fit_kwargs = {"verbose": False}
    predict_kwargs = {}
    metric_name = "accuracy"
    metric_list = ["accuracy", "roc_auc", "f1", "confusion_matrix"]
    metric_opt_dir_list = ["max", "max", "max", "compr"]
    metric_kwargs = {}
    artifact_directory = "model_artifacts/HPT"
    n_trials = 500
    if_callback = True
    cat_feat_fit: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {})


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="HPT_config_schema", node=HPT_Config)
