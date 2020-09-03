from enum import Enum


class MessageType(Enum):
    GradientPush = 0
    ParameterPull = 1
    ParameterPush = 2
    PoisonPill = 3