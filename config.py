
NN_PROJECT_PATH = "/home/matej/git/bandos/run.py"
NN_PROJECT_CONFIG_PRIMARY_PATH = "/home/matej/git/bandos/config.py"
NN_PROJECT_CONFIG_SECONDARY_PATH = "/etc/bandos/config.py"  # Secondary overrides primary
NN_LOG_FILE = "/home/matej/git/bandos/logs/default.txt"

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