from types import ModuleType
import os


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
    PARAMS_TO_OPT = "PARAMS_TO_OPT"
    LOG_TAGS_NAME = "LOG_TAGS"
    # check if NN config has PARAMS_TO_OPT
    common.are_params_to_opt(parsed_config.NN_PROJECT_CONFIG_PRIMARY_PATH,
                             parsed_config.NN_PROJECT_CONFIG_SECONDARY_PATH,
                             PARAMS_TO_OPT,
                             LOG_TAGS_NAME
                             )
    # run script
    common.run_script(parsed_config.NN_PROJECT_PATH)
    # change NN config
