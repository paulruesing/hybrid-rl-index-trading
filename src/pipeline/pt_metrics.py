import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class HitRateMetric(nn.Module):
    def __init__(self):
        """ Hit Rate Metric. Checks how often predictions point into the correct direction of targets. """
        super().__init__()

    def forward(self, predictions, targets, features):
        """ Callable. Requires predictions, targets AND features to work. """
        # convert to tensors (shape is [batch_size, sequence_length]):
        if not isinstance(predictions, torch.Tensor): predictions = torch.Tensor(predictions)
        if not isinstance(targets, torch.Tensor): targets = torch.Tensor(targets)
        if not isinstance(features, torch.Tensor): features = torch.Tensor(features)

        # compare direction of prediction and targets:
        hit_rate_mask = (torch.sign(predictions[:, -1]
                                    - features[:, -1]) == torch.sign(targets[:, -1]
                                                                    - features[:,-1]))  # true if last prediction and last target point in same direction starting from last feature
        return hit_rate_mask.sum() / hit_rate_mask.numel()  # hit rate as ratio


class WeightedMSELoss(nn.Module):
    def __init__(self, step_weights=None):
        """ Weighted MSE Loss. If step_weights is None, weights all predictions equally. """
        super().__init__()
        self.step_weights = step_weights

    def forward(self, predictions, targets):
        # predictions and target shape: [batch_size, n_steps]
        # convert to tensor dtype:
        if not isinstance(predictions, torch.Tensor): predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets)
        # calculate loss:
        if self.step_weights is None:  # equal weighting
            return torch.mean((predictions - targets) ** 2)  # MSE formula
        else:  # weighted loss
            losses = (predictions - targets) ** 2
            weighted_losses = losses * torch.Tensor(self.step_weights,
                                                    device=predictions.device)  # calculate weighted loss tensor
            return torch.mean(weighted_losses)