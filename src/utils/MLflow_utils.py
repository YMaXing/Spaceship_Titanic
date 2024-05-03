from mlflow.tracking import MlflowClient


def list_all_experiments() -> None:
    client = MlflowClient()
    experiments = client.search_experiments()  # This should list all experiments

    for exp in experiments:
        print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}, Lifecycle Stage: {exp.lifecycle_stage}")


def get_experiment_ID(experiment_name: str) -> None:
    # Initialize the MLflow client
    client = MlflowClient()

    # Retrieve the existing experiment by name
    experiment = client.get_experiment_by_name("Untuned_Models")

    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with ID: {experiment_id}")
        return experiment_id
    else:
        print("Experiment not found, consider creating a new one.")
        return None
