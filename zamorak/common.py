import os, re
import subprocess
from typing import Optional
from optuna.trial._trial import Trial

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


def run_script(script_path: str, python_path: str) -> int:
    """
    Runs a Python script at the given path and waits until it has completed.

    :param script_path: The file path to the Python script to be executed.

    :return: int The exit code of the process. Typically, an exit code of 0 indicates success.
    """
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


def load_config_as_dict(config_path: str) -> dict:
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


def are_params_to_opt(primary_path: str,
                      secondary_path: str,
                      optim_param_name: str,
                      log_tags_name: str) -> tuple[st.OptimizeParams, st.LogTags]:
    # check in both configs if exists param PARAMS_TO_OPT
    nn_config: dict = load_config_as_dict(primary_path)
    nn_config.update(load_config_as_dict(secondary_path))  # load .env

    # find PARAM_TO_OPT
    if not nn_config.get(optim_param_name):
        raise f"No {optim_param_name} found"
    # find LOG_TAGS
    if not nn_config.get(log_tags_name):
        raise f"No {log_tags_name} found"
    return st.OptimizeParams(**nn_config.get(optim_param_name)), st.LogTags(**nn_config.get(log_tags_name))


def find_value_in_log(nn_log_path: str, accuracy_tag: str, difference_tag: str) -> tuple[float, float]:
    """
    Searches for float or int values in a log file immediately following specified tags.
    """
    difference_value: Optional[float] = None
    accuracy_value: Optional[float] = None

    # Compile regex patterns to find float or int values after the specified tags
    difference_pattern = re.compile(rf"{re.escape(difference_tag)}\s*([+-]?\d+\.?\d*)")
    accuracy_pattern = re.compile(rf"{re.escape(accuracy_tag)}\s*([+-]?\d+\.?\d*)")

    try:
        with open(nn_log_path, 'r') as file:
            for line in file:
                # Search for difference value
                if difference_value is None:
                    match = difference_pattern.search(line)
                    if match:
                        difference_value = float(match.group(1))

                # Search for accuracy value
                if accuracy_value is None:
                    match = accuracy_pattern.search(line)
                    if match:
                        accuracy_value = float(match.group(1))

                # Break the loop if both values are found
                if difference_value is not None and accuracy_value is not None:
                    break

    except FileNotFoundError:
        log.error(f"Error: The file at '{nn_log_path}' does not exist.")
    except Exception as e:
        log.error(f"An error occurred while reading the file: {str(e)}")

    if not difference_value:
        log.error(f"not found average_loss_tag {accuracy_tag}")
        raise ValueError("not found difference_value: ")
    if not accuracy_value:
        log.error(f"not found accuracy_value {accuracy_value}")
        raise ValueError("not found accuracy_value: ")
    return difference_value, accuracy_value


def get_log_values(script_path: str,  python_path: str, nn_log_path: str, log_tags: st.LogTags) -> tuple[float, float]:
    run_script(script_path, python_path)
    average_loss_value, difference = find_value_in_log(nn_log_path, log_tags.accuracy.tag, log_tags.difference.tag)

    return average_loss_value, difference


def change_env_file(config_path: str, values_to_change: dict[str, any]) -> None:
    """
    Modifies a configuration file at the specified path. Parameters found in the file are updated,
    and those not found are added to the end of the file.

    :param: config_path (str): The file path to the configuration file.
    :param: values_to_change (dict[str, any]): A dictionary of parameter names and their new values.
    """
    create_file_if_not_exists(config_path)

    with open(config_path, 'r') as file:
        lines = file.readlines()

    # Create a map from existing lines to easily update/add new values
    config_dict = {}
    for line in lines:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            config_dict[key.strip()] = value.strip()

    # Update the dictionary with new values
    config_dict.update(values_to_change)
    with open(config_path, 'w') as file:
        for key, value in config_dict.items():
            file.write(f"{key}={value}\n")


def objective(trial: Trial,
              python_path: str,
              nn_project_path: str,
              nn_log_file_path: str,
              nn_secondary_config_path: str,
              optimize_params: st.OptimizeParams,
              log_tags: st.LogTags):
    # Suggest values for the hyperparameters
    margin: float = trial.suggest_float(optimize_params.margin.name,
                                 optimize_params.margin.boundary.min_value,
                                 optimize_params.margin.boundary.max_value)
    learning_rate: float = trial.suggest_float(optimize_params.learning_rate.name,
                                        optimize_params.learning_rate.boundary.min_value,
                                        optimize_params.learning_rate.boundary.max_value)
    # change config
    change_env_file(nn_secondary_config_path, {
                                                    optimize_params.margin.name: margin,
                                                    optimize_params.learning_rate.name: learning_rate,
                                                  })
    # Simulate running a script that uses these parameters and returns metrics
    # Replace `get_log_values` with your actual function to retrieve metrics
    # For the purpose of this example, let's assume it returns a tuple (accuracy, difference)
    accuracy, difference = get_log_values(nn_project_path,
                                          python_path,
                                          nn_log_file_path,
                                          log_tags)
    return difference, accuracy


