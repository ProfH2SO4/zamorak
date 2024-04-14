import os
import subprocess

from . import log


def create_file_if_not_exists(path_to_file: str) -> None:
    """
    Checks if a file exists at the specified path, and if not, creates the file along with any necessary directories.

    :param path_to_file: The full path to the file that needs to be checked and potentially created.
    :return: None. The function's purpose is to ensure the file exists, not to return any value.
    """
    directory = os.path.dirname(path_to_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(path_to_file):
        with open(path_to_file, "w") as file:
            pass  # Create an empty file


def run_script(script_path: str) -> int:
    """
    Runs a Python script at the given path and waits until it has completed.

    :param script_path: The file path to the Python script to be executed.

    :return: int The exit code of the process. Typically, an exit code of 0 indicates success.
    """
    python_path: str = "/home/matej/git/zezima/venv/bin/python3.10"
    original_dir: str = os.getcwd()  # Save the current working directory
    script_dir: str = os.path.dirname(script_path)  # Get the directory of the script

    try:
        os.chdir(script_dir)  # Change to the script directory
        # Run the script and wait for it to complete
        result = subprocess.run([python_path, script_path], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        # Handle cases where the script exits with an error
        r: str = f"Script failed with return code {e.returncode}"
        log.error(r)
        return e.returncode
    except FileNotFoundError:
        # Handle cases where the python interpreter or script is not found
        r = f"Script or Python interpreter not found at provided path: {script_path}"
        log.warning(r)
        return -1
    finally:
        os.chdir(original_dir)  # Restore the original directory


def load_config_from_py(config_path: str) -> dict:
    # Initialize a dictionary to execute the file within a safe scope
    local_vars = {}
    try:
        # Read the content of the config file
        with open(config_path, 'r') as file:
            config_content = file.read()

        # Execute the config content in an isolated local_vars dictionary
        exec(config_content, {}, local_vars)

        # Filter out any built-in entries or methods to clean up the namespace
        return {k: v for k, v in local_vars.items() if not k.startswith('__')}

    except FileNotFoundError:
        log.debug(f"Error: The file at '{config_path}' does not exist.")
        return {}
    except Exception as e:
        log.warning(f"An error occurred while loading the config: {str(e)}")
        return {}


def are_params_to_opt(primary_path: str, secondary_path: str, optim_param_name: str, log_tags_name: str) -> None:
    # check in both configs if exists param PARAMS_TO_OPT
    nn_config: dict = load_config_from_py(primary_path)
    nn_config.update(load_config_from_py(secondary_path))

    # find PARAM_TO_OPT
    if not nn_config.get(optim_param_name):
        raise f"No {optim_param_name} found"
    # find LOG_TAGS
    if not nn_config.get(log_tags_name):
        raise f"No {log_tags_name} found"


def optimize_config(primary_path: str, secondary_path: str):
    # check in both configs if exists param PARAMS_TO_OPT
    PARAMS_TO_OPT = "PARAMS_TO_OPT"
    nn_config: dict = load_config_from_py(primary_path)
    nn_config.update(load_config_from_py(secondary_path))

    # find PARAM_TO_OPT
    if not nn_config.get(PARAMS_TO_OPT):
        raise f"No {PARAMS_TO_OPT} found"
