import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fl1 = nn.Linear(2, 64)
        self.fl2 = nn.Linear(64, 128)
        self.fl3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):

        x = self.fl1(x)
        x = F.relu_(x)
        x = self.fl2(x)
        x = F.relu_(x)
        x = self.fl3(x)
        x = F.relu_(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)

        return x
