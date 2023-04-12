from typing import Type
from enum import Enum


class InstanceSelectorChooser(Enum):
    HARDEXAMPLEMINING = "hard_example_mining"
    SELFPACELEARNING = "self_paced_learning"
    ACTIVEBIAS = "active_bias"
    RANDOM = "random"

    def __str__(self):
        return self.value
    
    def __setstate__(self, state):
        print("setstate", state)
