
#
PYTHON_PATH = "/home/matej/git/bandos/venv/bin/python3.10"

NN_PROJECT_PATH = "/home/matej/git/bandos/run.py"
NN_PROJECT_CONFIG_PRIMARY_PATH = "/home/matej/git/bandos/config.py"
NN_PROJECT_ENV_FILE_PATH = "/home/matej/git/bandos/.env"  # Secondary overrides primary
NN_LOG_FILE = "/home/matej/git/bandos/logs/1k_hg19.txt"
NN_PARAM_CONFIG_NAME = "NAME"

# param names in config to know what we are looking for
NN_PARAMS_TO_OPT_NAME = "PARAMS_TO_OPT"
NN_LOG_TAGS_NAME = "LOG_TAGS"

# number of iteration to find optional params
N_TRIALS = 10


LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "ZAMORAK - %(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "sys_logger6": {
            "level": "DEBUG",
            "class": "logging.handlers.SysLogHandler",
            "formatter": "default",
            "address": "/dev/log",
            "facility": "local6",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",  # Use standard output
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "./logs/default.txt",  # Specify the file path
        },
    },
    "loggers": {
        "default": {
            "level": "DEBUG",
            "handlers": ["sys_logger6", "console", "file"],
            "propagate": False,
        }
    },
    "disable_existing_loggers": False,
}
