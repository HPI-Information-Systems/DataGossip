import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionTime(nn.Module):
    def __init__(self, seq_len: int = 1000, n_classes: int = 10) -> None:
        super().__init__()

        self.inception_block_1 = InceptionBlock()
        self.inception_block_2 = InceptionBlock()

        self.gap = nn.AvgPool1d(seq_len)

        self.fc = nn.Linear(10, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.inception_block_1(x)
        x = self.inception_block_2(x)
        
        x = self.gap(x).view(batch_size, -1)

        x = self.fc(x)

        return x


class InceptionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.inceptions = nn.ModuleList([Inception() for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_timeseries = x

        x = self.inceptions[0](x)
        x = self.inceptions[1](x)
        x = self.inceptions[2](x + input_timeseries)

        return x


class Inception(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)



