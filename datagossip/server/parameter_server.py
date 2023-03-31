import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from typing import List
from datetime import datetime
from ctypes import c_bool
import tqdm

from ..utils.distributed.messages import MessageListener, ModelSerializer, MessageSender
from ..utils.distributed.messages.type import MessageType
from ..utils.experiments import Experiment


def resize_data(data: torch.Tensor, args, size: int = 224):
    if args.model not in ["small", "medium", "large"]:
        data = nn.functional.interpolate(data, size=size)
    return data


#@torch.no_grad()
def test(model: nn.Module, data_loader: DataLoader, args):
    model.eval()
    correct = 0
    for data, target in tqdm.tqdm(data_loader):
        output = model(data)
        output = output.mean(dim=2)
        pred = output.max(1)[1]
        correct += pred.eq(target).sum().item()
    acc = correct / len(data_loader.dataset)
    model.train()
    return acc


class GradientPushListener(MessageListener):
    def set_message_type(self) -> MessageType:
        return MessageType.GradientPush

    def receive_message(self, sender: int):
        ModelSerializer.add_grads(self.model, self.receive_buffer)


class ParameterPullListener(MessageListener):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_sender = MessageSender()

    def build_receive_buffer(self) -> torch.Tensor:
        return torch.empty(1)

    def set_message_type(self) -> MessageType:
        return MessageType.ParameterPull

    def receive_message(self, sender: int):
        self.message_sender(MessageType.ParameterPush, ModelSerializer.flatten_model(self.model, grads=False), sender)


class ModelTester(mp.Process):
    def __init__(self, model: nn.Module, test_loader: DataLoader, args):
        super().__init__()
        self.daemon = True
        self.model = model
        self.dataloader = test_loader
        self.args = args
        self.is_running = mp.Value(c_bool, True)

    def stop(self):
        with self.is_running.get_lock():
            self.is_running.value = False

    def run(self) -> None:
        e = 0
        start_time = datetime.now()
        experiment = Experiment(".", metrics=["acc", "process_time"], attributes=dict(self.args._get_kwargs()))
        experiment._add_experiment()
        experiment.results = experiment._load_results()
        while self.is_running.value:
            copied_model = torch.nn.Conv1d(in_channels=1, out_channels=7, kernel_size=1)
            if self.dataloader is not None:
                test_acc = test(copied_model, self.dataloader, self.args)
            else:
                copied_model(torch.rand(1, 1, 20))
                test_acc = 0
            time = (datetime.now() - start_time).seconds
            experiment.add_results(e, test_acc, time)
            e += 1
            experiment._commit_results()


class ParameterServer:
    def __init__(self, model: nn.Module, group: dist.group, client_ranks: List[int], args, test_loader: DataLoader = None, test_model: nn.Module = None):
        print("setup listeners")
        self.listeners = [
            GradientPushListener(model),
            ParameterPullListener(model)
        ]
        self.model_tester = None
        if test_loader is not None:
            #test_model.share_memory()
            self.model_tester = ModelTester(torch.nn.Conv1d(in_channels=1, out_channels=7, kernel_size=1), None, args)
        self.group = group
        self.client_ranks = client_ranks
        print("sync model")
        self._sync_model(model)
        self.is_running = False

    def _sync_model(self, model: nn.Module):
        flat_model = ModelSerializer.flatten_model(model, grads=False)
        dist.broadcast(flat_model, src=0, group=self.group)
        print("dist barrier")
        dist.barrier(group=self.group)

    def start(self):
        self.is_running = True
        for thread in self.listeners:
            thread.start()
        if self.model_tester is not None:
            self.model_tester.start()
        self._wait_for_kill()

    def _wait_for_kill(self):
        poison_pill = torch.empty(1)
        while len(self.client_ranks) > 0:
            rank = dist.recv(poison_pill, tag=MessageType.PoisonPill.value)
            self.client_ranks.pop(self.client_ranks.index(rank))
            with open("poisons.txt", "a") as f:
                f.write(f"{datetime.now()}\n")
            print(f"poison pill received from {rank}")
        self.is_running = False
        self.print_report()
        if self.model_tester is not None:
            self.model_tester.stop()
        print("waiting for model tester to finish")
        while self.model_tester.is_alive():
            pass
        print("model tester finished")
        dist.barrier(group=self.group)

    def print_report(self):
        for listener in self.listeners:
            print(f"Received \t {listener.counter} \t messages from \t {listener}")

    def is_alive(self) -> bool:
        return self.is_running
