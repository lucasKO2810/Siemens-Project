import torch
import torch.nn as nn


class WeightedCrossEntropy(torch.nn.Module):
    """
    Class for weighted binary cross entropy loss.
    """
    def __init__(self, weight):
        super().__init__()
        self.weights = weight

    def forward(self, output, target):
        ce_loss = nn.BCELoss(weight=self.weights)
        loss = ce_loss(output, target)
        if abs(output[0][1]-target[0][0]) < 0.5:
            loss += 1
        return loss
