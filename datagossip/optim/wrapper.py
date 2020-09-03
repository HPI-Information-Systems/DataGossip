import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from .downpour_sgd import DownpourListener
from ..utils.distributed import ModelSerializer, MessageSender, MessageType


def downpour_wrapper(optimizer: Optimizer.__class__) -> Optimizer.__class__:
    class DownpourOptimizer(optimizer):
        def __init__(self, parameters, lr: float, model: torch.nn.Module, n_pull: int, n_push: int, group: dist.group,
                     node_parallelization: bool = False):
            super().__init__(parameters, lr=lr)
            self.iteration = 0
            self.n_pull = n_pull
            self.n_push = n_push
            self.model = model
            self.accgrad = torch.zeros_like(ModelSerializer.flatten_model(self.model, grads=False))
            self.parallel = node_parallelization
            if node_parallelization:
                self.accgrad.share_memory_()
            self.group = group

            # todo - only send classification layer of pretrained models
            self.push_message_sender = MessageSender()
            self.pull_message_sender = MessageSender()
            self.message_listener = DownpourListener(self.model)
            if not self.parallel:
                self.message_listener.start()
                self._sync_model()

        def setup_parallel(self):
            self.message_listener.start()
            self._sync_model()

        def _sync_model(self):
            dist.broadcast(self.accgrad, src=0, group=self.group)
            ModelSerializer.overwrite_params(self.model, self.accgrad)
            self.accgrad.zero_()
            dist.barrier(group=self.group)

        def _push_gradients(self):
            if self.push_message_sender(MessageType.GradientPush, self.accgrad):
                self.accgrad.zero_()

        def _pull_model(self):
            self.pull_message_sender(MessageType.ParameterPull, torch.empty(1))

        def kill_master(self):
            dist.send(torch.empty(1), dst=0, tag=MessageType.PoisonPill.value)
            dist.barrier(group=self.group)

        def _get_params(self) -> torch.Tensor:
            before = torch.zeros_like(self.accgrad)
            offset = 0
            for group in self.param_groups:
                for p in group['params']:
                    length = offset + p.flatten().shape[0]
                    before[offset:length].add_(p.flatten())
                    offset = length
            return before

        @torch.no_grad()
        def step(self, closure=None):
            before = self._get_params()
            loss = super().step(closure)
            after = self._get_params()

            self.accgrad.add_(after - before)

            if self.iteration % self.n_push == 0:
                self._push_gradients()

            if self.iteration % self.n_pull == 0:
                self._pull_model()

            self.iteration += 1

            return loss

    return DownpourOptimizer
