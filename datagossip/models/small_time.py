import torch.nn as nn
import torch.nn.functional as F


class SmallTimeModel(nn.Module):
    def __init__(self, in_channels=1, seq_len=1000, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3)

        self.max_pool = nn.MaxPool1d(2)
        self.conv2_drop = nn.Dropout()

        neurons = ((((((seq_len - 4) // 2) - 4) // 2) - 2) // 2) * 20
        self.fc1 = nn.Linear(neurons, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = F.relu(self.max_pool(self.conv2_drop(self.conv3(x))))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    model = SmallTimeModel()

    # Test the model
    import torch
    x = torch.rand(1, 1, 1000)
    y = model(x)
    print(y.shape)