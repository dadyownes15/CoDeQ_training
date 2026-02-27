import torch
import torch.nn as nn



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
 