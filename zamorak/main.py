from types import ModuleType
import os
import optuna
from functools import partial

import config

from . import common
from . import log
from . import struct as st


def load_config(path_: str) -> ModuleType:
    """
    Loads the configuration from a local `config.py` file.
     If a `config.py` file exists in `/etc/bandos/`, it overrides parameters in the local `config.py`.

    :return: The configuration module loaded with settings from the configuration file.
    """
    app_config: ModuleType = config

    if not os.path.isfile(path_):
        return app_config
    try:
        with open(path_, "rb") as rnf:
            exec(compile(rnf.read(), path_, "exec"), app_config.__dict__)
    except OSError as e:
        print(f"File at {path_} could not be loaded because of error: {e}")
        raise e from e
    return app_config


def parse_namespace(config_: ModuleType) -> st.Config:
    """
    Parses and filters the attributes of a configuration module, excluding any built-in attributes.

    :param config_: The configuration module to be parsed.
    :return: A dictionary containing the filtered configuration parameters.
    """
    parsed: dict[str, any] = {}
    for key, value in config_.__dict__.items():
        if not key.startswith("__"):
            parsed[key] = value
    return st.Config(**parsed)


def main() -> None:
    # load config
    config_: ModuleType = load_config("/etc/zamorak/config.py")
    parsed_config: st.Config = parse_namespace(config_)

    print("============ Setting Up Logger ============")
    if parsed_config.LOG_CONFIG["handlers"].get("file", None):
        file_path: str = parsed_config.LOG_CONFIG["handlers"]["file"].get("filename")
        common.create_file_if_not_exists(file_path)
    log.set_up_logger(parsed_config.LOG_CONFIG)
    log.info("START")
    log.info(f"Config Values: {parsed_config}")
    # check if NN config has PARAMS_TO_OPT
    optimize_param: st.OptimizeParams
    log_tag: st.LogTags
    optimize_param, log_tag = common.are_params_to_opt(
                             parsed_config.NN_PROJECT_CONFIG_PRIMARY_PATH,
                             parsed_config.NN_PROJECT_ENV_FILE_PATH,
                             parsed_config.NN_PARAMS_TO_OPT_NAME,
                             parsed_config.NN_LOG_TAGS_NAME
                             )

    # Create a multi-objective study #  common.objective => return difference, accuracy
    study = optuna.create_study(directions=[log_tag.difference.goal, log_tag.accuracy.goal])
    # nn_project_path: str, nn_log_file_path: str, log_tag
    wrapped_objective = partial(common.objective,
                                nn_project_path=parsed_config.NN_PROJECT_PATH,
                                nn_log_file_path=parsed_config.NN_LOG_FILE,
                                nn_secondary_config_path=parsed_config.NN_PROJECT_ENV_FILE_PATH,
                                optimize_params=optimize_param,
                                log_tags=log_tag)

    # Optimize the study, using the objective function and a callback for logging
    study.optimize(wrapped_objective, n_trials=parsed_config.N_TRIALS)

    log.debug("Best trials:")
    for trial in study.best_trials:
        log.debug(f"  Values: {trial.values}")
        log.debug(f"  Params: {trial.params}")
    log.info("END")
