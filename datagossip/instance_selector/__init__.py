from .base import InstanceSelector
from .active_bias import ActiveBias
from .self_paced_learning import SelfPacedLearning
from .hard_example_mining import HardExampleMining
from .random import Random
HARDEXAMPLEMINING = "hard_example_mining"
EASYEXAMPLEMINING = "easy_example_mining"
SELFPACELEARNING = "self_paced_learning"
ACTIVEBIAS = "active_bias"
RANDOM = "random"


def get_instance_selector_class(name: str):
    if name == RANDOM:
        return Random
    elif name == HARDEXAMPLEMINING:
        return HardExampleMining
    elif name == SELFPACELEARNING:
        return SelfPacedLearning
    elif name == ACTIVEBIAS:
        return ActiveBias
    else:
        raise ValueError(f"No IS method with name {name} exists")
