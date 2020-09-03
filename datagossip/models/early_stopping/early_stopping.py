import logging


class EarlyStopping:
    """Inspired by https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py"""

    def __init__(self,
                 patience: int,
                 min_delta: float = 0.0,
                 cumulative_delta: bool = False,
                 ):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            self.logger.debug(f"EarlyStopping: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.logger.info("EarlyStopping: Stop training")
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
