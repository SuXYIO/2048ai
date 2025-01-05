from torch import relu
import torch.nn as nn

class gamenn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 4)
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x
