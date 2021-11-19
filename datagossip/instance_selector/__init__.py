from typing import Type

from .base import InstanceSelector
from .active_bias import ActiveBias
from .self_paced_learning import SelfPacedLearning
from .hard_example_mining import HardExampleMining
from .random import Random
from enum import Enum


class InstanceSelectorChooser(Enum):
    HARDEXAMPLEMINING = "hard_example_mining"
    SELFPACELEARNING = "self_paced_learning"
    ACTIVEBIAS = "active_bias"
    RANDOM = "random"

    def get_instance_selector_class(self) -> Type[InstanceSelector]:
        if self == self.RANDOM:
            return Random
        elif self == self.HARDEXAMPLEMINING:
            return HardExampleMining
        elif self == self.SELFPACELEARNING:
            return SelfPacedLearning
        else:  # if self.name == self.ACTIVEBIAS:
            return ActiveBias

    def __str__(self):
        return self.value
