from .base import ForeignDataIterator


class ForeignCycleIterator(ForeignDataIterator):
    def _check_idx_pointer(self):
        if len(self.indices) <= self.idx_pointer:
            self.reset()
