from src.config_schemas.data_schemas.process_data_schema import Process_data_Config
from src.utils.config_utils import get_config
from src.utils.gcp_utils import access_secret_version


@get_config(config_path="../configs/data_configs", config_name="process_data_config")
def process_data(config: Process_data_Config) -> None:

    github_token = access_secret_version(
        config["data"].gcp_project_id, config["data"].gcp_secret_id, config["data"].gcp_version_id
    )

    print(f"github_token: {github_token}")


if __name__ == "__main__":
    process_data()  # type: ignore
