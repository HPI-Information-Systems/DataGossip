import torch
from torch.optim import SGD
import torch.distributed as dist
from ..utils.distributed import ModelSerializer, MessageListener, MessageSender, MessageType


class DownpourListener(MessageListener):
    def set_message_type(self) -> MessageType:
        return MessageType.ParameterPush

    def receive_message(self, sender: int):
        ModelSerializer.overwrite_params(self.model, self.receive_buffer)


class DownpourSGD(SGD):
    def __init__(self, parameters, lr: float, model: torch.nn.Module, n_pull: int, n_push: int, group: dist.group, node_parallelization: bool = False):
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

    @torch.no_grad()
    def step(self, closure=None):
        lr = self.param_groups[0]['lr']
        gradients = ModelSerializer.flatten_model(self.model, grads=True)
        self.accgrad.add_(gradients, alpha=-lr)

        if self.iteration % self.n_push == 0 and not self.parallel:
            self._push_gradients()

        loss = super().step(closure)

        if self.iteration % self.n_pull == 0 and not self.parallel:
            self._pull_model()

        self.iteration += 1

        return loss

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill_master()
