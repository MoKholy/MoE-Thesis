import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, expert_outputs, targets):
        ce_loss = self.ce_loss(expert_outputs, targets)
        return ce_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, expert_outputs, targets):
        logpt = -nn.functional.cross_entropy(expert_outputs, targets, reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -(1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[targets]
        return focal_loss.mean()

class MSEGatingLoss(nn.Module):
    def __init__(self):
        super(MSEGatingLoss, self).__init__()

    def forward(self, gating_outputs, true_gating_values):
        mse_loss = nn.MSELoss()(gating_outputs, true_gating_values)
        return mse_loss
