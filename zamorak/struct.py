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


@dataclass(slots=True)
class Boundary:
    min_value: float
    max_value: float


@dataclass(slots=True)
class Param:
    name: str
    boundary: Boundary


@dataclass(slots=True)
class OptimizeParams:
    learning_rate: Param
    margin: Param

    def __init__(self, **kwargs):
        self.learning_rate = Param(name="LEARNING_RATE", boundary=Boundary(**kwargs["LEARNING_RATE"]))
        self.margin = Param(name="MARGIN", boundary=Boundary(**kwargs["MARGIN"]))


@dataclass(slots=True)
class LogTags:
    average_loss: str
    accuracy: str
    accuracy_top_10: str
    difference: str

    def __init__(self, **kwargs):
        self.average_loss = kwargs["average_loss"]
        self.accuracy = kwargs["accuracy"]
        self.accuracy_top_10 = kwargs["accuracy_top_10"]
        self.difference = kwargs["difference"]
