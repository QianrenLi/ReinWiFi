# define different network
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class envNet(nn.Module):
    def __init__(self, actions, hidden_states, states, layers = 10):
        super(envNet, self).__init__()
        self.query = nn.Linear(actions, hidden_states)
        self.fc = self._make_layer(hidden_states, layers//2)
        self.fc1 = nn.Linear(hidden_states, 2)

        self.fc2 = nn.Linear(actions + 2, hidden_states)
        self.fc3 = self._make_layer(hidden_states, layers//2)
        self.fc4 = nn.Linear(hidden_states, states)

    def _make_layer(self, hidden_states, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_states, hidden_states))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        temp_q = self.query(x)
        temp_q = F.relu(temp_q)
        temp_q = self.fc(temp_q)
        temp_q = F.relu(temp_q)
        temp_q = self.fc1(temp_q)
        temp_q = F.relu(temp_q)
        temp_q = self.fc2(torch.cat((x, temp_q), dim = 1))
        temp_q = F.relu(temp_q)
        temp_q = self.fc3(temp_q)
        temp_q = F.relu(temp_q)
        temp_q = self.fc4(temp_q)
        return temp_q