import os, re
import subprocess
from typing import Optional

from . import log
from . import struct as st

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


def are_params_to_opt(primary_path: str, secondary_path: str,
                      optim_param_name: str, log_tags_name: str) -> tuple[st.OptimizeParams, st.LogTags]:
    # check in both configs if exists param PARAMS_TO_OPT
    nn_config: dict = load_config_from_py(primary_path)
    nn_config.update(load_config_from_py(secondary_path))

    # find PARAM_TO_OPT
    if not nn_config.get(optim_param_name):
        raise f"No {optim_param_name} found"
    # find LOG_TAGS
    if not nn_config.get(log_tags_name):
        raise f"No {log_tags_name} found"
    return st.OptimizeParams(**nn_config.get(optim_param_name)), st.LogTags(**nn_config.get(log_tags_name))


def find_value_in_log(nn_log_path: str, average_loss_tag: str, difference_tag: str) -> tuple[float, float]:
    """
    Searches for float or int values in a log file immediately following specified tags.
    """
    average_loss_value: Optional[float] = None
    accuracy_value: Optional[float] = None

    # Compile regex patterns to find float or int values after the specified tags
    average_loss_pattern = re.compile(rf"{re.escape(average_loss_tag)}\s*:\s*([+-]?\d*\.?\d+)")
    accuracy_pattern = re.compile(rf"{re.escape(difference_tag)}\s*:\s*([+-]?\d*\.?\d+)")

    try:
        with open(nn_log_path, 'r') as file:
            for line in file:
                # Search for average loss
                if average_loss_value is None:  # Continue searching until found
                    match = average_loss_pattern.search(line)
                    if match:
                        average_loss_value = float(match.group(1))

                # Search for accuracy
                if accuracy_value is None:  # Continue searching until found
                    match = accuracy_pattern.search(line)
                    if match:
                        accuracy_value = float(match.group(1))

                # Break the loop if both values are found
                if average_loss_value is not None and accuracy_value is not None:
                    break

    except FileNotFoundError:
        log.error(f"Error: The file at '{nn_log_path}' does not exist.")
    except Exception as e:
        log.error(f"An error occurred while reading the file: {str(e)}")

    if not average_loss_value:
        raise f"not found average_loss_value: {average_loss_value}"
    if not accuracy_value:
        raise f"not found accuracy_value: {accuracy_value}"
    return average_loss_value, accuracy_value


def get_log_values(script_path: str, nn_log_path: str, log_tags: st.LogTags) -> tuple[float, float]:
    run_script(script_path)
    average_loss_value, difference = find_value_in_log(nn_log_path, log_tags.average_loss, log_tags.difference)

    return average_loss_value, difference


