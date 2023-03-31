import torch
import torch.nn as nn
from typing import List, Tuple
from .type import MessageType
from copy import deepcopy


class ModelSerializer:
    @staticmethod
    def flatten_model(model: nn.Module, grads: bool = True) -> torch.Tensor:
        if grads:
            return torch.cat(list(p.grad.view(-1) for p in model.parameters()))
        return torch.cat(list(p.data.view(-1) for p in model.parameters()))

    @staticmethod
    @torch.no_grad()
    def add_grads(model: nn.Module, flat_tensor: torch.Tensor):
        offset = 0
        for p in model.parameters():
            inter_length = p.flatten().shape[0]
            p[:].add_(flat_tensor[offset:offset+inter_length].view(p.shape))
            offset += inter_length

    @staticmethod
    def overwrite_params(model: nn.Module, flat_tensor: torch.Tensor):
        offset = 0
        for p in model.parameters():
            inter_length = p.flatten().shape[0]
            p.data.copy_(flat_tensor[offset:offset + inter_length].view(p.shape))
            offset += inter_length

    @staticmethod
    def copy_model(model: nn.Module) -> nn.Module:
        new_model = deepcopy(model)
        return new_model
