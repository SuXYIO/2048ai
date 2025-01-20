import torch as tch
import torch.nn as nn

class gamefnn0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class gamefnn1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    if input('Export network templates?[y/n]: ') == 'y':
        network_name = input('Network class name: ')
        tch.save(eval(network_name), f'./templates/{network_name}_template')
        print(f'Exported to ./templates/{network_name}_template')
