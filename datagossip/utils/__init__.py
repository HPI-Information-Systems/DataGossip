import torch.multiprocessing as mp


def set_start_method(method="spawn"):
    start_method = mp.get_start_method()
    print(f"start method: {start_method}")
    if start_method != method:
        mp.set_start_method(method)