# firewall_model.py

import torch.nn as nn
import torch.nn.functional as F

class FirewallModel(nn.Module):
    def __init__(self, input_size):
        super(FirewallModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
