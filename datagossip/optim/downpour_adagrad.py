import torch
from torch.optim.adagrad import Adagrad
import torch.distributed as dist
from .downpour_sgd import DownpourListener
from ..utils.distributed import ModelSerializer, MessageListener, MessageSender, MessageType


class DownpourAdagrad(Adagrad):
    def __init__(self, parameters, lr: float, model: torch.nn.Module, n_pull: int, n_push: int, group: dist.group, node_parallelization: bool = False, parameter_server: bool = True):
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
        self.parameter_server = parameter_server

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
        if not self.parameter_server:
            return
        dist.broadcast(self.accgrad, src=0, group=self.group)
        ModelSerializer.overwrite_params(self.model, self.accgrad)
        self.accgrad.zero_()
        dist.barrier(group=self.group)

    def _push_gradients(self):
        if not self.parameter_server:
            return
        if self.push_message_sender(MessageType.GradientPush, self.accgrad):
            self.accgrad.zero_()

    def _pull_model(self):
        if not self.parameter_server:
            return
        self.pull_message_sender(MessageType.ParameterPull, torch.empty(1))

    def kill_master(self):
        dist.send(torch.empty(1), dst=0, tag=MessageType.PoisonPill.value)
        dist.barrier(group=self.group)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        offset = 0

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(p, alpha=group['weight_decay'])
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group['eps'])
                    p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    state['sum'].addcmul_(grad, grad, value=1)
                    std = state['sum'].sqrt().add_(group['eps'])
                    p.addcdiv_(grad, std, value=-clr)
                    length = offset+p.flatten().shape[0]
                    self.accgrad[offset:length].addcdiv_(grad.flatten(), std.flatten(), value=-clr)
                    offset = length

        if self.iteration % self.n_push == 0:
            self._push_gradients()

        if self.iteration % self.n_pull == 0:
            self._pull_model()

        self.iteration += 1

        return loss
