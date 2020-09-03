import torch
import torch.distributed as dist
from threading import Thread
from .type import MessageType


class MessageSender:
    def __init__(self):
        self.pending_messages = dict()

    def __call__(self, message_type: MessageType, tensor: torch.Tensor, dst: int = 0) -> bool:
        isend_thread = self.pending_messages.get(dst, None)
        if isend_thread is None or not isend_thread.is_alive():
            req = dist.isend(tensor, dst=dst, tag=message_type.value)
            isend_thread = Thread(target=MessageSender.request_thread, args=(req,))
            isend_thread.daemon = True
            isend_thread.start()
            self.pending_messages[dst] = isend_thread
            return True
        return False

    @staticmethod
    def request_thread(req):
        req.wait()
