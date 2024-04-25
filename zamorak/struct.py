from dataclasses import dataclass, fields
from optuna.study import StudyDirection


@dataclass(slots=True)
class Config:
    NN_PROJECT_PATH: str
    NN_PROJECT_CONFIG_PRIMARY_PATH: str
    NN_PROJECT_ENV_FILE_PATH: str  # Secondary overrides primary
    NN_LOG_FILE: str
    LOG_CONFIG: dict
    NN_PARAMS_TO_OPT_NAME: str
    NN_LOG_TAGS_NAME: str
    N_TRIALS: int
    PYTHON_PATH: str

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
class TagEntity:
    tag: str
    goal: StudyDirection

    def __init__(self, **kwargs):
        self.tag = kwargs["tag"]
        self.goal = self.convert_goal(kwargs["goal"])

    @staticmethod
    def convert_goal(goal_str: str) -> StudyDirection:
        """Converts a string to a StudyDirection."""
        if goal_str.upper() == "MINIMIZE":
            return StudyDirection.MINIMIZE
        elif goal_str.upper() == "MAXIMIZE":
            return StudyDirection.MAXIMIZE
        elif goal_str.upper() == "NOT_SET":
            return StudyDirection.NOT_SET
        else:
            raise ValueError(f"Unknown goal direction: {goal_str}")


@dataclass(slots=True)
class LogTags:
    average_loss: TagEntity
    accuracy: TagEntity
    accuracy_top_10: TagEntity
    difference: TagEntity

    def __init__(self, **kwargs):
        self.average_loss = TagEntity(**kwargs["average_loss"])
        self.accuracy = TagEntity(**kwargs["accuracy"])
        self.accuracy_top_10 = TagEntity(**kwargs["accuracy_top_10"])
        self.difference = TagEntity(**kwargs["difference"])
