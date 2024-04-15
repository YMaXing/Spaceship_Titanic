from src.utils.utils import run_shell_command
from shutil import rmtree


def get_cmd_to_get_raw_data(version: str,
                            data_local_save_dir: str,
                            dvc_remote_repo: str,
                            dvc_data_folder: str,
                            github_user_name: str,
                            github_access_token: str) -> str:
    """
    Get shell command to download raw data from dvc store

    Parameters
    ----------
    version : str
        version of the data
    data_local_save_dir : str
        local directory to save the data
    dvc_remote_repo : str
        remote dvc repo holding data information
    dvc_data_folder : str
        dvc folder where data is stored
    github_user_name : str
        github user name
    github_access_token : str
        github access token

    Returns
    -------
    str
        shell command to download raw data from dvc store
    """
    dvc_remote_repo = f"https://{github_user_name}:{github_access_token}@{dvc_remote_repo}"
    command = f"dvc get {dvc_remote_repo} {dvc_data_folder} --rev {version} -o {data_local_save_dir}"

    return command


def get_raw_data_with_version(version: str,
                              data_local_save_dir: str,
                              dvc_remote_repo: str,
                              dvc_data_folder: str,
                              github_user_name: str,
                              github_access_token: str) -> None:
    rmtree(data_local_save_dir, ignore_errors=True)
    command = get_cmd_to_get_raw_data(version, data_local_save_dir, dvc_remote_repo, dvc_data_folder, github_user_name, github_access_token)
    run_shell_command(command)
