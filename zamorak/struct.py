from dataclasses import dataclass, fields


@dataclass(slots=True)
class Config:
    NN_PROJECT_PATH: str
    NN_PROJECT_CONFIG_PRIMARY_PATH: str
    NN_PROJECT_CONFIG_SECONDARY_PATH: str  # Secondary overrides primary
    NN_LOG_FILE: str
    LOG_CONFIG: dict

    def __init__(self, **kwargs):
        cls_fields = {f.name for f in fields(self)}
        for key in kwargs:
            if key in cls_fields:
                setattr(self, key, kwargs[key])