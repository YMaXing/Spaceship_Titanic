import logging
from pathlib import Path
from src.utils.config_utils import get_config
from src.config_schemas.config_schema import Config
import mlflow
import pandas as pd
from src.utils.HPT_utils import read_data


@get_config(config_path="../configs", config_name="config")
def main(cfg: Config):
    # Create the final data directory
    final_directory = Path(cfg.final_data_folder) / f"{cfg.final_model_name}_{cfg.final_encoding}"
    if not final_directory.exists():
        final_directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Successfully created directory {final_directory}")
    else:
        logging.info(f"Directory {final_directory} already exists")

    # Load the final model after hyperparameter tuning
    final_model_run_id = cfg.final_model_run_id
    final_model = mlflow.sklearn.load_model(final_model_run_id)

    # Load the final test data
    df_train, df_test = read_data(cfg.local_test_data_dir + f"/{cfg.final_encoding}")
    logging.info(f"Successfully loaded test data from {cfg.local_test_data_dir}/{cfg.final_encoding}")

    # Train the final model on the training data without cross-validation
    final_model_no_cv = final_model.estimator(**final_model.estimators[0].get_params())
    final_model_no_cv.fit(df_train.drop(cfg.label, axis=1), df_train[cfg.label], **final_model.fit_kwargs)
    no_vote_prediction = final_model_no_cv.predict(df_test, **final_model.predict_kwargs).astype(bool)
    # Make predictions on the test data, both hard and soft voting from all the estimators trained on the cv folds of the training data
    hard_vote_prediction, soft_vote_prediction = final_model.predict(df_test, majority_threshold=cfg.majority_threshold)
    hard_vote_prediction = hard_vote_prediction.astype(bool)
    soft_vote_prediction = soft_vote_prediction.astype(bool)
    logging.info("Predictions made successfully")

    # Load the sample submission file
    no_vote_sample_submission = pd.read_csv(cfg.local_sample_submission_dir)
    no_vote_sample_submission[cfg.label] = no_vote_prediction
    hard_vote_sample_submission = pd.read_csv(cfg.local_sample_submission_dir)
    hard_vote_sample_submission[cfg.label] = hard_vote_prediction
    soft_vote_sample_submission = pd.read_csv(cfg.local_sample_submission_dir)
    soft_vote_sample_submission[cfg.label] = soft_vote_prediction
    # Save the predictions
    no_vote_sample_submission.to_csv(final_directory / "no_vote_submission.csv", index=False)
    hard_vote_sample_submission.to_csv(final_directory / "hard_vote_submission.csv", index=False)
    soft_vote_sample_submission.to_csv(final_directory / "soft_vote_submission.csv", index=False)
    logging.info("Predictions saved successfully.")


if __name__ == "__main__":
    main()
