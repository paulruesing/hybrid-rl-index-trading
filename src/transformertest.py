import src.utils.file_management as filemgmt
import src.pipeline.preprocessing as prep
import src.pipeline.predictors as predictors
from src.pipeline.predictors import LSTMPredictor, TransformerPredictor

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Union

from tqdm import tqdm
from alpha_vantage.timeseries import TimeSeries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

ROOT = Path().resolve().parent

DATA = ROOT / "data"
DAILY_PRICES = DATA / "daily_price_downloads"
MINUTELY_PRICES = DATA / "minutely_price_downloads"
INTERPOLATED_PRICES = DATA / "interpolated_prices_dax"
#INTERPOLATED_SMOOTHED_PRICES = DATA / "interpolated_smoothed_prices_dax"

SAVED_MODELS = DATA / "saved_models" / "smoothed_data"


if __name__ == "__main__":
    transformer_predictor = TransformerPredictor(
        # data parameters:
        price_csv_path=filemgmt.most_recent_file(INTERPOLATED_PRICES, '.csv', 'at 7d'),
        daily_prediction_hour=16,  # these are necessary data properties which still need to be set
        rolling_window_size=6 * 4,  # i.e. 6 months á 4 weeks
        forecast_horizon=3,  # i.e. 3 weeks
        sampling_rate_minutes=7 * 14 * 60,  # 1 week = 7 days each from 8am to 22pm
        validation_split=.2,

        # model parameters:
        init_weights=True,
        hidden_transformer_layer_size=512,
        n_transformer_layers=6,
        n_transformer_heads=1,
        forecast_step_loss_weight_range=(.7, 1.0),
        use_pre_transformer_fc_layer=True,
        dropout=.4
    )

    transformer_predictor.run_training(custom_n_epochs=100, custom_early_stopping_patience=10,
                                       visualise_validation_predictions_every=None)