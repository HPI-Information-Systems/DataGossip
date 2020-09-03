import yaml
import os
import subprocess
import fire
from itertools import product
from typing import List
from copy import deepcopy

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


def get_experiments_temp(settings):
    exp = deepcopy(settings)
    for k, v in settings.items():
        if type(v) != list:
            exp[k] = v
    for datagossip in settings.get("datagossip"):
        exp["datagossip"] = datagossip
        exp["optimizer"] = "sgd" if datagossip else "adagrad"
        exp["batch_size"] = settings.get("batch_size")[2]
        exp["lr"] = settings.get("lr")[3]
        exp["n_push_pull"] = settings.get("n_push_pull")[1]
        yield exp

        for i, batch_size in enumerate(settings.get("batch_size")):
            if i == 2:
                continue
            exp["batch_size"] = batch_size
            yield exp
        exp["batch_size"] = settings.get("batch_size")[2]

        for i, lr in enumerate(settings.get("lr")):
            if i == 3:
                continue
            exp["lr"] = lr
            yield exp
        exp["lr"] = settings.get("lr")[3]

        for i, n_push_pull in enumerate(settings.get("n_push_pull")):
            if i == 1:
                continue
            exp["n_push_pull"] = n_push_pull
            yield exp


def transform_args(items) -> List[str]:
    args = []
    for k, v in items:
        if k == "repetitions":
            continue
        args.append(f"--{k}={v}")

    return args


def change_settings(settings: dict) -> dict:
    settings_path = os.path.join(DIRECTORY, "large.yml")
    with open(settings_path, "r") as f:
        alexnet = yaml.load(f, Loader=yaml.FullLoader)
    datagossip = "datagossip" if settings.get("datagossip") else "baseline"
    imbalanced = "imbalanced" if settings.get("imbalanced") else "balanced"
    special = alexnet[datagossip][imbalanced]
    settings["lr"] = special["lr"]
    settings["n_push_pull"] = special["n_push_pull"]
    settings["optimizer"] = special["optimizer"]
    return settings


def run(rank=0, size=2, python="python"):
    settings = load_yaml()
    repetitions = settings.get("args", {}).get("repetitions", 1)
    for _ in range(repetitions):
        for experiment in get_combinations(settings.get("args", {})):
            subprocess.call([python, os.path.join(DIRECTORY, "train.py"), f"--rank={rank}", f"--size={size}", *transform_args(experiment.items())])


def get_command(rank, size, python="python"):
    settings = load_yaml()
    for experiment in get_combinations(settings.get("args", {})):
        yield " ".join([python, os.path.join(DIRECTORY, "local_experiments.py"), f"--rank={rank}", f"--size={size}",
                         *[f"--{k}={v}" for k, v in experiment.items() if k != "repetitions"]])


if __name__ == "__main__":
    fire.Fire(run)
