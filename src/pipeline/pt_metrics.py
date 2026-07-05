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
    def __init__(self, step_weights=None, target_scale: float = 100.0):
        """
        Weighted MSE Loss. If step_weights is None, weights all predictions equally.

        Parameters
        ----------
        step_weights : array-like, optional
            Per-forecast-step weights. If None, weights all predictions equally.
        target_scale : float, default 100.0
            Scaling factor applied to (predictions - targets) before squaring. Since predictions and
            targets are ratio-space values (Phase 1.1; e.g. ~1.0 +/- a few percent), the raw MSE lives
            in the 1e-6 float-noise range and produces gradients close to Adam's eps=1e-9. This scaling
            is a constant factor (target_scale ** 2) on the MSE -- it does not change relative loss
            comparisons (early stopping's strict '<', ReduceLROnPlateau's scale-invariant relative
            threshold_mode) -- but it lifts logged/filename losses and gradients into a readable range.
            This scaling must live only here, inside the loss -- never applied to the data/model
            outputs/HitRateMetric, which all remain in unscaled ratio space.
        """
        super().__init__()
        self.target_scale = target_scale
        # convert once here (not every forward call) and .to(device) below in forward -- this also
        # fixes a latent bug in the old code, which called the legacy `torch.Tensor(data, device=...)`
        # constructor every forward pass; that constructor rejects non-CPU devices and would have
        # crashed the moment use_mps_if_available=True or CUDA was used.
        self.step_weights = None if step_weights is None else torch.tensor(step_weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        # predictions and target shape: [batch_size, n_steps]
        # convert to tensor dtype:
        if not isinstance(predictions, torch.Tensor): predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets)
        # scale ratio-space difference up before squaring -- mathematically equivalent to
        # ((p - 1) * target_scale - (t - 1) * target_scale) == (p - t) * target_scale:
        scaled_diff = (predictions - targets) * self.target_scale
        losses = scaled_diff ** 2
        # calculate loss:
        if self.step_weights is None:  # equal weighting
            return torch.mean(losses)  # MSE formula
        else:  # weighted loss
            weighted_losses = losses * self.step_weights.to(losses.device)  # calculate weighted loss tensor
            return torch.mean(weighted_losses)