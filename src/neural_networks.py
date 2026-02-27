import torch
import torch.nn as nn

class CifarConvMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 32
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc2 = nn.Linear(8 * 14 * 14, self.width)
        self.fc3 = nn.Linear(self.width, self.width)
        self.fc4 = nn.Linear(self.width, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)

        # MLP layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class CifarMLP(nn.Module):
    """Simple 3-layer MLP for CIFAR-10: 3072 → 120 → 84 → 10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 32 * 32, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))



class MLP_small_fashion_mnist(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.net(x)  
        

class tiny_test_model(nn.Module):
    def __init__(self, remove_column) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.ReLU()
        )

        if (remove_column):
            self._zero_middle_neuron()

    def _zero_middle_neuron(self):
        hidden = self.net[2]  # nn.Linear(3, 3)
        with torch.no_grad():
            hidden.weight[:, 1] = 0.0

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
 