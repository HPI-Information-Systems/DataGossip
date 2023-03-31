import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionNetwork(nn.Module):
    def __init__(self, in_channels: int, seq_len: int = 1000, out_channels: int = 7, bottleneck_size: int = 32) -> None:
        super().__init__()

        self.inception_block_1 = InceptionBlock(in_channels, bottleneck_size)
        self.inception_block_2 = InceptionBlock(bottleneck_size*4, bottleneck_size)

        self.gap = nn.AvgPool1d(seq_len)

        self.fc = nn.Linear(10, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.inception_block_1(x)
        x = self.inception_block_2(x)
        
        x = self.gap(x).view(batch_size, -1)

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, input_channels: int, bottleneck_size: int = 32) -> None:
        super().__init__()

        self.inceptions = nn.ModuleList([Inception(input_channels, bottleneck_size) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_timeseries = x

        x = self.inceptions[0](x)
        x = self.inceptions[1](x)
        x = self.inceptions[2](x + input_timeseries)

        return x


class Inception(nn.Module):
    def __init__(self, input_channels: int, bottleneck_size: int = 32) -> None:
        super().__init__()

        self.conv = nn.Conv1d(input_channels, bottleneck_size, kernel_size=1, padding='same')
        self.max_pooling = nn.MaxPool1d(kernel_size=3, stride=1, padding='same')

        self.conv_1 = nn.Conv1d(bottleneck_size, bottleneck_size, kernel_size=10, padding='same')
        self.conv_2 = nn.Conv1d(bottleneck_size, bottleneck_size, kernel_size=20, padding='same')
        self.conv_3 = nn.Conv1d(bottleneck_size, bottleneck_size, kernel_size=40, padding='same')
        self.conv_bottleneck = nn.Conv1d(input_channels, bottleneck_size, kernel_size=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.conv(x)

        x_1 = self.conv_1(bottleneck)
        x_2 = self.conv_2(bottleneck)
        x_3 = self.conv_3(bottleneck)
        x_4 = self.conv_bottleneck(self.max_pooling(x))

        x = torch.cat([x_1, x_2, x_3, x_4], dim=2)  # todo: which dim to concat?
        x = F.batch_norm(x)
        x = F.relu(x)
        return x
