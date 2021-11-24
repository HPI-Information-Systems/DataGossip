import yaml
import os
import subprocess
import fire
from itertools import product
from typing import List

EXPERIMENTS_SETTINGS = "experiment_settings.yml"
DIRECTORY = os.path.join(os.path.abspath("."), "experiments")


def load_yaml() -> dict:
    settings_path = os.path.join(DIRECTORY, EXPERIMENTS_SETTINGS)
    with open(settings_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings


def get_combinations(settings):
    exp = dict()
    keys = []
    lists = []
    for k, v in settings.items():
        if type(v) == list:
            lists.append(v)
            keys.append(k)
        else:
            exp[k] = v

    for combination in product(*lists):
        for k, v in zip(keys, combination):
            exp[k] = v
        yield exp


def transform_args(items) -> List[str]:
    args = []
    for k, v in items:
        if k == "repetitions":
            continue
        args.append(f"--{k}={v}")

    return args


def run(rank=0, size=2, python="python"):
    settings = load_yaml()
    repetitions = settings.get("args", {}).get("repetitions", 1)
    for _ in range(repetitions):
        for experiment in get_combinations(settings.get("args", {})):
            subprocess.call([python, os.path.join(DIRECTORY, "train.py"), f"--rank={rank}", f"--size={size}", *transform_args(experiment.items())])


if __name__ == "__main__":
    fire.Fire(run)
