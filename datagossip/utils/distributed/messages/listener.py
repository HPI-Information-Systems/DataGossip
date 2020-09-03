import torch
import torch.distributed as dist
from threading import Thread
from abc import abstractmethod
from .model_serializer import ModelSerializer
from .type import MessageType


class MessageListener(Thread):
    def __init__(self, model: torch.nn.Module):
        self.counter = 0
        self.model = model
        self.message_type = self.set_message_type()
        self.receive_buffer = self.build_receive_buffer()
        self.is_running = False
        super().__init__()
        self.daemon = True

    def build_receive_buffer(self) -> torch.Tensor:
        return torch.zeros(ModelSerializer.flatten_model(self.model, grads=False).shape[0])

    @abstractmethod
    def set_message_type(self) -> MessageType:
        pass

    @abstractmethod
    def receive_message(self, sender: int):
        pass

    def run(self):
        self.is_running = True
        while self.is_running:
            sender = dist.recv(self.receive_buffer, tag=self.message_type.value)
            self.counter += 1
            self.receive_message(sender)

    def __str__(self):
        return self.__class__.__name__
