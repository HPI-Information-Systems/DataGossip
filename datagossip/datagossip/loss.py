import torch
from torch.nn import NLLLoss
import torch.nn.functional as F

from ..instance_selector.base import InstanceSelector


class DataGossipLoss(NLLLoss):
    def __init__(self, instance_selector: InstanceSelector, loss_fn=F.nll_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instance_selector = instance_selector
        self.loss_fn = loss_fn

    def forward(self, input, target):
        single_losses = self.loss_fn(input, target, reduction="none", weight=self.weight)
        if self.instance_selector.local_training_:
            self.instance_selector.update(single_losses.detach())
        return single_losses.mean()
