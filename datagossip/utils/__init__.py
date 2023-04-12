import torch.multiprocessing as mp


def set_start_method(method="spawn"):
    try:
        mp.set_start_method(method)
    except RuntimeError:
        pass