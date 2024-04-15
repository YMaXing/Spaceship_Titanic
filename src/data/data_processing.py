from src.utils.config_utils import get_config
from src.config_schemas.data.data_processing_config_schema import data_processing_Config
from src.utils.data_utils import get_raw_data_with_version
from src.utils.gcp_utils import access_secret_version


@get_config(config_path="../configs/data", config_name="data_processing_config")
def process_data(config: data_processing_Config) -> None:

    github_access_token = access_secret_version(config.gcp_project_id, config.gcp_secret_id, config.version)
    get_raw_data_with_version(version=config.version,
                              data_local_save_dir=config.data_local_save_dir,
                              dvc_remote_repo=config.dvc_remote_repo,
                              dvc_data_folder=config.dvc_data_folder,
                              github_user_name=config.github_user_name,
                              github_access_token=github_access_token)


if __name__ == "__main__":
    process_data()  # type: ignore
