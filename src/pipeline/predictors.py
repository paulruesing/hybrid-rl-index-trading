import pandas as pd
import numpy as np
from typing import Literal, Union, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import src.utils.file_management as filemgmt
import src.pipeline.preprocessing as preprocessing
import src.pipeline.pt_metrics as metrics


class TimeSeriesDataset(Dataset):
    """ Time-series dataset class based on torch.utils.data.Dataset compatible torch.utils.data.DataLoader """
    def __init__(self,
                 x: Union[pd.DataFrame, pd.Series, np.ndarray],
                 y: Union[pd.Series, pd.Series, np.ndarray],
                 verbose=False):
        # convert values to numpy if necessary:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        # expand tensor for LSTM input w. shape (batch, sequence_length, input_size):
        x = np.expand_dims(x, 2)

        # sanity check:
        if x.shape[0] != y.shape[0]: raise ValueError("Mismatched number of samples.")

        # save values:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

        # information:
        if verbose: print(f"Dataset Shape: {self.x.shape}, {self.y.shape}")

    # necessary operators:
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TransformerModel(nn.Module):
    """
    Transformer model class with attention pooling and learnable positional encoding.

    Parameters
    ----------
    input_size : int, default=1
        Number of features per input time step.
    hidden_layer_size : int, default=32
        Hidden size of transformer embeddings.
    num_layers : int, default=2
        Number of transformer encoder layers.
    n_forecast_steps : int, default=1
        Number of steps to forecast (output sequence length).
    dropout : float, default=0.2
        Dropout rate inside transformer layers.
    n_heads : int, default=4
        Number of attention heads in transformer.
    use_pre_transformer_fc_layer : bool, default=False
        Whether to apply a linear + ReLU layer before transformer.
    max_seq_len : int, default=100
        Maximum sequence length expected (for positional encoding).
    init_weights : bool, default=True
        Whether to initialize weights manually.
    """

    def __init__(self,
                 input_size: int = 1,
                 hidden_layer_size: int = 32,
                 num_layers: int = 2,
                 n_forecast_steps: int = 1,
                 dropout: float = 0.2,
                 n_heads: int = 4,
                 use_pre_transformer_fc_layer: bool = False,
                 max_seq_len: int = 100,
                 init_weights: bool = True):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.use_pre_transformer_fc_layer = use_pre_transformer_fc_layer
        self.n_forecast_steps = n_forecast_steps
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.linear_1 = nn.Linear(input_size, hidden_layer_size) if use_pre_transformer_fc_layer else None
        self.relu = nn.ReLU() if use_pre_transformer_fc_layer else None

        # transformer input features:
        d_model = hidden_layer_size if use_pre_transformer_fc_layer else input_size

        # Learnable positional encoding parameter:
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=hidden_layer_size * 2,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.Linear(d_model, 1)

        self.linear_2 = nn.Linear(d_model, n_forecast_steps)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        """
        Initialize model weights and biases.

        Weights are initialized with Xavier uniform distribution.
        Biases are initialized to zero.
        Positional encoding is initialized with a small normal distribution.

        Returns
        -------
        None
        """
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        # Initialize positional encoding with small values
        nn.init.normal_(self.positional_encoding, mean=0, std=0.01)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, n_forecast_steps).
        """
        if self.use_pre_transformer_fc_layer:
            temp_x = self.linear_1(x)
            temp_x = self.relu(temp_x)
        else:
            temp_x = x

        seq_len = temp_x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Add positional encoding (slice to sequence length)
        temp_x = temp_x + self.positional_encoding[:, :seq_len, :]

        transformer_out = self.transformer_encoder(temp_x)

        attn_scores = self.attention_pool(transformer_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled_output = (transformer_out * attn_weights).sum(dim=1)

        predictions = self.linear_2(pooled_output)

        # TODO: consider more sophisticated decoding schemes

        return predictions.squeeze(-1)

    def run_epoch(self, dataloader, optimiser, device='cpu', loss_criterion=nn.MSELoss(), is_training=False):
        """
        Run one epoch of training or validation.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader providing batches of (inputs, targets).
        optimiser : torch.optim.Optimizer
            Optimizer to update model parameters during training.
        device : str or torch.device, default 'cpu'
            Device on which to run computations.
        loss_criterion : torch.nn.Module, default nn.MSELoss()
            Loss function to compute training/validation loss.
        is_training : bool, default False
            Whether to perform training (True) or evaluation (False).

        Returns
        -------
        tuple
            Tuple containing:
            - epoch_loss (float): Sum of batch losses across the epoch.
            - lr (float): Current learning rate of the optimizer.
        """
        epoch_loss = 0
        self.train() if is_training else self.eval()

        for idx, (x, y) in enumerate(dataloader):
            if is_training:
                optimiser.zero_grad()

            x = x.to(device)
            y = y.to(device)

            out = self(x)
            loss = loss_criterion(out.contiguous(), y.contiguous())

            if is_training:
                loss.backward()
                optimiser.step()

            epoch_loss += loss.detach().item()

        lr = optimiser.param_groups[0]['lr']
        return epoch_loss, lr

    def predict(self, dataloader, device='cpu'):
        """
        Generate predictions for all samples in the dataloader.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader providing batches of inputs (and optionally targets).
        device : str or torch.device, default 'cpu'
            Device on which to run computations.

        Returns
        -------
        numpy.ndarray
            Array of model predictions for all samples.
        """
        self.eval()
        temp_loader = DataLoader(dataloader.dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(temp_loader)):
                x = x.to(device)
                out = self(x)
                out = out.cpu().detach().numpy()
                if idx == 0:
                    predictions = np.array(out)
                    continue
                predictions = np.concatenate((predictions, out))

            return predictions


class LSTMModel(nn.Module):
    """
    LSTM model class based on torch.nn.Module.

    Parameters
    ----------
    input_size : int, default=1
        Number of features per input time step.
    hidden_layer_size : int, default=32
        Hidden size of LSTM layers.
    num_layers : int, default=2
        Number of stacked LSTM layers.
    n_forecast_steps : int, default=1
        Number of time steps to forecast. If greater than 1, predictions are recursively fed back as inputs.
    dropout : float, default=0.2
        Dropout rate applied between LSTM layers.
    use_pre_lstm_fc_layer : bool, default=False
        Whether to use a fully connected layer before LSTM to transform input features.
    use_final_hidden_state : bool, default=True
        Whether to use the final hidden state of all LSTM layers as output or last LSTM output step.
    use_hidden_states_across_forecast_steps : bool, default=True
        Whether to pass hidden and cell states across recursive forecast steps to reduce computation.
    init_weights : bool, default=True
        Whether to initialize weights with custom initialization.
    """
    def __init__(self,
                 input_size=1,
                 hidden_layer_size=32,
                 num_layers=2,
                 n_forecast_steps=1,  # output sequence length
                 dropout=0.2,
                 use_pre_lstm_fc_layer=False,
                 use_final_hidden_state=True,
                 use_hidden_states_across_forecast_steps=True,
                 init_weights=True):
        """
        Initialise the LSTM model.

        :param input_size: input size of the LSTM model.
        :param hidden_layer_size: hidden size of the LSTM model.
        :param num_layers: number of LSTM layers.
        :param n_forecast_steps: number of forecast steps. If not 1, outputs will be calculated recursively for each forecast step, i.e. while feeding back previous predictions into the network.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.use_pre_lstm_fc_layer = use_pre_lstm_fc_layer
        self.use_hidden_states_across_forecast_steps = use_hidden_states_across_forecast_steps
        self.n_forecast_steps = n_forecast_steps

        # optional pre-lstm fully connected layer:
        self.linear_1 = nn.Linear(input_size,
                                  hidden_layer_size) if use_pre_lstm_fc_layer else None  # transform feature into size of lstm hidden layers
        self.relu = nn.ReLU() if use_pre_lstm_fc_layer else None  # non-linearity

        # lstm layers:
        self.lstm = nn.LSTM(hidden_layer_size if use_pre_lstm_fc_layer else 1, hidden_size=self.hidden_layer_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)  # batch first=True because our shape is (batch, seq, feature)
        # self.dropout = nn.Dropout(dropout)  # prevent overfitting, now is included in lstm

        self.use_final_hidden_state = use_final_hidden_state  # use lstm_out, else use h_n; see comments below in forward()

        # final fully connected layer:
        self.linear_2 = nn.Linear((num_layers if use_final_hidden_state else 1) * hidden_layer_size,
                                  1)  # transforms LSTM output into 1 price
        # when use_final_hidden_state, multiply with * num_layers to utilise all LSTM layer's final hidden states

        if init_weights: self.init_weights()  # empirically accelerates convergence

    def init_weights(self):
        """
        Initialize model weights and biases.

        Weights for input-hidden connections are initialized with Kaiming normal initialization.
        Weights for hidden-hidden connections are initialized with orthogonal initialization.
        Biases are initialized to zero.

        Returns
        -------
        None
        """
        for name, param in self.named_parameters():  # iterate through weights and biases
            if 'bias' in name:  # bias zero initialisation: neutral starting point
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:  # input-hidden weights with He/Kaiming initialisation w. normal distribution: preserves gradient magnitude through ReLU activations
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:  # hidden-hidden weights with orthogonal matrix initialisation: maintains norm during sequence processing and helps prevent gradient issues
                nn.init.orthogonal_(param)

    def forward(self, x):
        """
        Forward pass of the model.

        Recursively predicts `n_forecast_steps` future time steps.
        Optionally uses hidden and cell states across recursive steps to reduce computation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_forecast_steps).
        """
        batchsize = x.shape[0]

        outputs = torch.empty(size=(batchsize, self.n_forecast_steps), device=x.device,
                              dtype=x.dtype)  # initialise empty tensor for outputs

        # recursive prediction (taking into account previous predictions if n_forecast_steps > 1):
        if self.use_hidden_states_across_forecast_steps:
            h_n, c_n = None, None  # initialise hidden cell states that will be used across steps to reduce computational complexity
            # this is favourable, because otherwise the complete sequence is reprocessed each iteration
            # and LSTMs are designed to remember previous steps through hidden states
        for output_sequence_ind in range(self.n_forecast_steps):
            # integrate previous predictions into input:
            if output_sequence_ind > 0:
                x = torch.roll(x, shifts=-1)  # shift all inputs back by one time unit
                x[:, -1] = torch.unsqueeze(outputs[:, output_sequence_ind - 1],
                                           1)  # replace most recent time step with last prediction

            if self.use_pre_lstm_fc_layer:
                # layer 1:
                temp_x = self.linear_1(
                    x)  # we need to denote this as temp_x because we need the original input again in latter iterations
                # x = self.batch_norm(x)
                temp_x = self.relu(temp_x)
                # x.shape now is (batch, sequenth_length, features)
            else: temp_x = x

            # LSTM layer(s):
            if self.use_hidden_states_across_forecast_steps:
                # pass last hidden states again to lstm and compute only most recent step
                if h_n is None:  # first iteration
                    lstm_out, (h_n, c_n) = self.lstm(temp_x)  # compute first step and get hidden and cell states
                else:  # further iterations
                    lstm_out, (h_n, c_n) = self.lstm(temp_x[:, -1].unsqueeze(1), (
                    h_n, c_n))  # compute new information and leverage previous hidden and cell states
            else:
                lstm_out, (h_n, c_n) = self.lstm(temp_x)

            # which output to use?
            # - lstm_out contains **all hidden states for the last layer** for every time step, used for tasks requiring per-time-step predictions and **sequence forecasting**
            #       shape is (batchsize, sequence_length, features)
            # - h_n is a tensor containing the **final hidden state for each layer** in the sequence, used for initialing latter sequences, as summarisation for classification tasks or **one-step-forecasting**
            #       shape is (num_layers, batchsize, features)
            # - c_n is a tensor containing the **final cell state for each layer** in the sequence, used jointly with h_n for more complex forecasting purposes
            #       shape is (num_layers, batchsize, features)
            if self.use_final_hidden_state:
                # we utilise h_n because it aggregates states from all layers.
                # reshape final hidden state output from (num_layers, batchsize, features) to (batchsize, features) for consecutive linear layer:
                # permute rearranges the dimensions according to a specific order, -1 tells reshape to infer the correct size for that dimension
                temp_x = h_n.permute(1, 0, 2).reshape(batchsize, -1)  # here we flatten all layer outputs
            else:
                # we utilise lstm_out[:, -1, :]  which is the last time step of the final layer
                temp_x = lstm_out[:, -1, :]

            # layer 2:
            # x = self.dropout(x)  additional regularisation is spared here because happening inside LSTM
            predictions = self.linear_2(temp_x)
            outputs[:, output_sequence_ind] = predictions.squeeze(
                -1)  # return last column of all batches (equals predictions), squeeze removes dimensions with size 1

        return outputs.squeeze(-1)  # remove dimensions with size 1

    def run_epoch(self, dataloader, optimiser, device='cpu', loss_criterion=nn.MSELoss(), is_training=False):
        """
        Run one epoch of training or validation.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader providing batches of (inputs, targets).
        optimiser : torch.optim.Optimizer
            Optimizer to update model parameters during training.
        device : str or torch.device, default 'cpu'
            Device on which to run computations.
        loss_criterion : torch.nn.Module, default nn.MSELoss()
            Loss function to compute training/validation loss.
        is_training : bool, default False
            Whether to perform training (True) or evaluation (False).

        Returns
        -------
        tuple
            Tuple containing:
            - epoch_loss (float): Sum of batch losses across the epoch.
            - lr (float): Current learning rate of the optimizer.
        """
        epoch_loss = 0

        if is_training:
            self.train()  # training mode activates Dropout and BatchNorm (updates running statistics with each batch)
        else:
            self.eval()  # evaluation mode for inference or testing

        # iterate through dataset's batches via provided dataloader instance:
        for idx, (x, y) in enumerate(dataloader):
            if is_training: optimiser.zero_grad()  # reset all gradients for next training step

            # move tensors to correct device:
            x = x.to(device)
            y = y.to(device)

            # compute output and loss:
            out = self(x)
            loss = loss_criterion(out.contiguous(),
                                  y.contiguous())  # enforces the tensors to be stored in a contiguous memory block

            if is_training:
                loss.backward()  # backpropagation, traverses computational graph in reverse applying the chain rule to compute gradients
                optimiser.step()  # optimise weights

            epoch_loss += loss.detach().item()  # without / batchsize because loss is already averaged, detach loss value from computational graph

        # learning rate:
        lr = optimiser.param_groups[0]['lr']

        return epoch_loss, lr

    def predict(self, dataloader, device='cpu'):
        """
        Generate predictions for all samples in the dataloader.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader providing batches of inputs (and optionally targets).
        device : str or torch.device, default 'cpu'
            Device on which to run computations.

        Returns
        -------
        numpy.ndarray
            Array of model predictions for all samples.
        """
        self.eval()  # inference mode

        # deactivate shuffling and set batch size to 1:
        temp_loader = DataLoader(dataloader.dataset, batch_size=1, shuffle=False)

        # compute predictions:
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(temp_loader)):
                x = x.to(device)
                out = self(x)  # run model
                out = out.cpu().detach().numpy()
                if idx == 0: predictions = np.array(out); continue  # initialise array upon first iteration
                predictions = np.concat((predictions, out))

            return predictions


class NNPredictor:
    def __init__(self,
                 sampling_rate_minutes: int = 15,  # data import parameters
                 price_csv_path: str = None,
                 price_column: str = 'close',
                 date_column: str = 'date',
                 daily_prediction_hour: int = None,  # data preparation parameters
                 predict_before_daily_prediction_hour: bool = False,
                 rolling_window_size: int = 32,
                 forecast_horizon: int = 12,
                 validation_split: float = 0.2,
                 randomise_validation: bool = True,
                 batch_size: int = 32,

                 forecast_step_loss_weight_range: [float, float] = (1, 0.7),  # eval parameter
                 use_mps_if_available: bool = False,  # device parameter

                 # training parameters:
                 model_save_directory: str = None,
                 n_train_epochs: int = None,  # if set to some number, trains upon initialisation
                 lr_scheduler: Literal['step', 'plateau'] = 'plateau',
                 initial_lr: float = 0.001,
                 step_scheduler_step_size: int = 40,
                 plateau_scheduler_factor: float = 0.5,
                 early_stopping_patience: int = 30,  # if 0 doesn't utilise early stopping
                 evaluate_hit_rate_upon_training=True,  # can save run-time especially for short trainig procedures
                 verbose: bool = False):
        ### provided parameters:
        # data preparation:
        self._sampling_rate_minutes = sampling_rate_minutes
        self._price_csv_path = price_csv_path
        self._date_column = date_column
        self._price_column = price_column
        self._daily_prediction_hour = daily_prediction_hour
        self._predict_before_daily_prediction_hour = predict_before_daily_prediction_hour
        self._rolling_window_size = rolling_window_size
        self._forecast_horizon = forecast_horizon
        self._validation_split = validation_split
        self.randomise_validation = randomise_validation
        self._batch_size = batch_size
        self.verbose = verbose
        self._use_mps_if_available = use_mps_if_available
        self._forecast_step_loss_weight_range = forecast_step_loss_weight_range

        # training:
        self.model_save_directory = model_save_directory
        self._n_train_epochs = n_train_epochs
        self._lr_scheduler = lr_scheduler
        self._initial_lr = initial_lr
        self._step_scheduler_step_size = step_scheduler_step_size
        self._plateau_scheduler_factor = plateau_scheduler_factor
        self._early_stopping_patience = early_stopping_patience
        self.evaluate_hit_rate_upon_training = evaluate_hit_rate_upon_training

        ### placeholders:
        self._price_series = self._normalised_price_series = None
        self._normaliser = preprocessing.Normaliser()  # initialise normaliser
        _ = self.normalised_price_series  # access property once, leads to fitting of normaliser for further use
        self._X = self._Y = self._X_dates = self._Y_dates = None
        self._X_train = self._X_val = self._Y_train = self._Y_val = self._X_dates_train = self._X_dates_val = self._Y_dates_train = self._Y_dates_val = None
        self._dataloader_train = self._dataloader_val = None

        self._device = None

        self._predictions_train = None
        self._predictions_val = None

        self._loss_criterion = None
        self._loss_train = self._loss_val = None
        self._hit_rate_train = self._hit_rate_val = None

        ### status messages:
        if price_csv_path is None and verbose: print(
            'No price file provided yet. Define price_csv_path to trigger data import.')
        if daily_prediction_hour is None and verbose: print(
            'No daily prediction hour defined yet, hence currently predictions are carried out at every time step. Define daily_prediction_hour to change this.')

    # data import parameters:
    @property
    def sampling_rate_minutes(self):
        return self._sampling_rate_minutes

    @property
    def price_csv_path(self):
        return self._price_csv_path

    @price_csv_path.setter
    def price_csv_path(self, value):
        self._price_csv_path = value;
        self.import_data()

    @property
    def price_column(self):
        return self._price_column

    @price_column.setter
    def price_column(self, value):
        self._price_column = value
        if self._price_series is not None: self._price_series.name = value

    @property
    def date_column(self):
        return self._date_column

    @date_column.setter
    def date_column(self, value):
        self._date_column = value
        if self._price_series is not None: self._price_series.index.name = value

    @property
    def price_series(self):
        if self._price_series is None:
            if self.price_csv_path is None:
                raise AttributeError("LSTMPredictor.price_csv_path needs to be provided to import price data.")
            else:
                self.import_data()
        return self._price_series

    ### data preparation properties:
    @property
    def normalised_price_series(self):
        if self._normalised_price_series is None:
            self._normalised_price_series = self._normaliser.fit_transform(self.price_series)
        return self._normalised_price_series

    @property
    def normaliser(self):
        return self._normaliser

    @property
    def daily_prediction_hour(self):
        return self._daily_prediction_hour

    @daily_prediction_hour.setter
    def daily_prediction_hour(self, value):
        self._daily_prediction_hour = value;
        self.prepare_data()

    @property
    def predict_before_daily_prediction_hour(self):
        return self._predict_before_daily_prediction_hour

    @predict_before_daily_prediction_hour.setter
    def predict_before_daily_prediction_hour(self, value):
        self._predict_before_daily_prediction_hour = value
        # re-prepare data if sampling rate is below 1 hour, because point of prediction changes
        if self.sampling_rate_minutes < 60: self.prepare_data()

    @property
    def rolling_window_size(self):
        return self._rolling_window_size

    @rolling_window_size.setter
    def rolling_window_size(self, value):
        self._rolling_window_size = value;
        self.prepare_data()

    @property
    def forecast_horizon(self):
        return self._forecast_horizon

    @forecast_horizon.setter
    def forecast_horizon(self, value):
        self._forecast_horizon = value;
        self.prepare_data()

    @property
    def X(self):
        if self._X is None: self.prepare_data()
        return self._X

    @property
    def Y(self):
        if self._Y is None: self.prepare_data()
        return self._Y

    @property
    def X_dates(self):
        if self._X_dates is None: self.prepare_data()
        return self._X_dates

    @property
    def Y_dates(self):
        if self._Y_dates is None: self.prepare_data()
        return self._Y_dates

    @property
    def validation_split(self):
        return self._validation_split

    @validation_split.setter
    def validation_split(self, value):
        self._validation_split = value;
        self.split_data()

    @property
    def X_train(self):
        if self._X_train is None: self.split_data()
        return self._X_train

    @property
    def X_val(self):
        if self._X_val is None: self.split_data()
        return self._X_val

    @property
    def Y_train(self):
        if self._Y_train is None: self.split_data()
        return self._Y_train

    @property
    def Y_val(self):
        if self._Y_val is None: self.split_data()
        return self._Y_val

    @property
    def X_dates_train(self):
        if self._X_dates_train is None: self.split_data()
        return self._X_dates_train

    @property
    def X_dates_val(self):
        if self._X_dates_val is None: self.split_data()
        return self._X_dates_val

    @property
    def Y_dates_train(self):
        if self._Y_dates_train is None: self.split_data()
        return self._Y_dates_train

    @property
    def Y_dates_val(self):
        if self._Y_dates_val is None: self.split_data()
        return self._Y_dates_val

    @property
    def dataset_train(self):
        return TimeSeriesDataset(self.X_train, self.Y_train)

    @property
    def dataset_val(self):
        return TimeSeriesDataset(self.X_val, self.Y_val)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dataloader_train(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    @property
    def dataloader_val(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=True)

    ### training properties:
    @property
    def use_mps_if_available(self):
        return self._use_mps_if_available

    @property
    def device(self):
        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device(
                    'cuda')  # if ROCm is installed and AMD hardware is supported by such then 'cuda' also refers to AMD GPU acceleration. This however is not possible on mac
                if self.verbose: print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
            elif torch.backends.mps.is_available() and self.use_mps_if_available:
                self._device = torch.device('mps')
                if self.verbose: print("Using GPU through Metal Performance Shaders (MPS) backend")
            else:
                self._device = torch.device('cpu')
                if self.verbose: print("Using CPU")
        return self._device

    @property
    def n_train_epochs(self):
        return self._n_train_epochs

    @n_train_epochs.setter
    def n_train_epochs(self, value):
        self._n_train_epochs = value;
        self.run_training()

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def initial_lr(self):
        return self._initial_lr

    @property
    def step_scheduler_step_size(self):
        return self._step_scheduler_step_size

    @property
    def plateau_scheduler_factor(self):
        return self._plateau_scheduler_factor

    @property
    def early_stopping_patience(self):
        return self._early_stopping_patience

    ### prediction properties:
    @property
    def predictions_train(self):
        if self._predictions_train is None:
            self._predictions_train = self.nn_model.predict(self.dataloader_train, device=self.device)
        return self._predictions_train

    @property
    def predictions_val(self):
        if self._predictions_val is None:
            self._predictions_val = self.nn_model.predict(self.dataloader_val, device=self.device)
        return self._predictions_val

    ### evaluation properties:
    @property
    def forecast_step_loss_weight_range(self):
        return self._forecast_step_loss_weight_range

    @forecast_step_loss_weight_range.setter
    def forecast_step_loss_weight_range(self, value):
        """ Changing value re-initialises loss criterion. """
        self._forecast_step_loss_weight_range = value;
        self._loss_criterion = None

    @property
    def loss_criterion(self):
        """ Forecast step loss weighted according to forecast_step_loss_weight_range. """
        if self._loss_criterion is None: self._loss_criterion = metrics.WeightedMSELoss(
            step_weights=np.linspace(self.forecast_step_loss_weight_range[0],
                                     self.forecast_step_loss_weight_range[1],
                                     self.forecast_horizon))
        return self._loss_criterion

    @property
    def loss_train(self):
        """ Final training loss value """
        return self.loss_criterion(self.predictions_train, self.Y_train)

    @property
    def loss_val(self):
        """ Final validation loss value """
        return self.loss_criterion(self.predictions_val, self.Y_val)

    @property
    def hit_rate_train(self):
        """ How often model predicts right direction of price development (hit rate) in training samples. """
        return metrics.HitRateMetric()(self.predictions_train, self.Y_train, self.X_train).item()

    @property
    def hit_rate_val(self):
        """ How often model predicts right direction of price development (hit rate) in validation samples. """
        return metrics.HitRateMetric()(self.predictions_val, self.Y_val, self.X_val)

    ### data preparation methods:
    def import_data(self):
        """ Import data from LSTMPredictor.price_csv_path file. """
        self._price_series = preprocessing.read_price_csv(csv_path=self._price_csv_path,
                                                          date_column=self._date_column,
                                                          price_column=self._price_column)
        self._predictions_val = self._predictions_train = None

    def prepare_data(self):
        """
        Leverages preprocessing.create_rolling_window_view().

        Creates a rolling window matrix of training data and target values based on a time-series with datetime index.
        Columns of training data are defined by rolling_window_size, columns of target data by forecast_horizon.

        For intra-day predictions, the method allows for defining the ending point of each rolling window (and
        hence starting point of target values) with daily_prediction_hour. E.g. daily_prediction_hour=15 will lead
        to the targets always starting at the first sample after 3 pm.
        This further requires specifying sampling_rate_minutes to find the first entry in that prediction hour.
        """
        (self._X, self._Y, self._X_dates, self._Y_dates
         ) = preprocessing.create_rolling_window_view(input_series=self.normalised_price_series,
                                                      rolling_window_size=self.rolling_window_size,
                                                      forecast_horizon=self.forecast_horizon,
                                                      sampling_rate_minutes=self.sampling_rate_minutes,
                                                      daily_prediction_hour=self.daily_prediction_hour,
                                                      predict_before_daily_prediction_hour=self.predict_before_daily_prediction_hour,
                                                      verbose=self.verbose)
        self._predictions_train = self._predictions_val = None  # and reset predictions

    def split_data(self, verbose: bool = None):
        """
        Leverages preprocessing.create_train_validation_split.

        Splits training and target values into training and validation split.
        """
        (self._X_train, self._X_val, self._Y_train, self._Y_val,
         self._X_dates_train, self._X_dates_val, self._Y_dates_train, self._Y_dates_val
         ) = preprocessing.create_train_validation_split(X=self.X,
                                                         Y=self.Y,
                                                         X_dates=self.X_dates,
                                                         Y_dates=self.Y_dates,
                                                         verbose=self.verbose if verbose is None else verbose,
                                                         validation_split=self.validation_split,
                                                         randomise=self.randomise_validation)
        self._predictions_train = self._predictions_val = None  # and reset predictions

    ### plotting methods:
    def plot_train_validation_overview(self):
        """ Plot training and validation data highlighted by colors. """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.Y_dates_train[:, 0], self._normaliser.inverse_transform(self.Y_train[:, 0]), label='Training Data',
                color='blue')  # use first column (first price in forecast sequence)
        ax.plot(self.Y_dates_val[:, 0], self._normaliser.inverse_transform(self.Y_val[:, 0]), label='Validation Data',
                color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Data Overview')
        ax.grid(True)
        ax.legend()

    def plot_prediction_overview(self, data_split: Literal['training', 'validation'] = 'validation',
                                 day_slice: (int, int) = None, custom_plot_title: str = None,
                                 X_color: str = 'blue', Y_color: str = 'red', pred_color: str = 'green',
                                 plot_size: (int, int) = (12, 6),
                                 ) -> None:
        """ Plot prediction overview on training or validation data. """
        predictions_per_day = 1

        # select training or validation data:
        predictions = self.predictions_train if data_split == 'training' else self.predictions_val
        X_dates = self.X_dates_train if data_split == 'training' else self.X_dates_val
        X = self.X_train if data_split == 'training' else self.X_val
        Y_dates = self.Y_dates_train if data_split == 'training' else self.Y_dates_val
        Y = self.Y_train if data_split == 'training' else self.Y_val

        # prepare day_slice:
        if day_slice is None or len(day_slice) != 2:
            day_slice = (0, len(predictions))

        # plot result for daily multi-step predictions:
        fig, ax = plt.subplots(figsize=plot_size)
        for ind, (x_datetime, features, y_datetime, pred, target) in enumerate(
                zip(X_dates, X, Y_dates, predictions, Y)):
            # plot only days within day_slice:
            if ind / predictions_per_day < day_slice[0]: continue
            if ind / predictions_per_day >= day_slice[1]: break  # plot only that many days
            # plot features:
            ax.plot(x_datetime, self._normaliser.inverse_transform(features), color=X_color)
            # plot prediction and target:
            ax.plot(y_datetime, self._normaliser.inverse_transform(target), color=Y_color, linewidth=3)
            ax.plot(y_datetime, self._normaliser.inverse_transform(pred), color=pred_color, linestyle='--')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Result Overview' if custom_plot_title is None else custom_plot_title)
        legend_elements = [Line2D([0], [0], color=X_color, label='Training Prices'),
                           Line2D([0], [0], color=Y_color, label='Target Prices', linewidth=3),
                           Line2D([0], [0], color=pred_color, label='Predicted Prices', linestyle='--'), ]
        ax.legend(handles=legend_elements)
        ax.grid(True)
        plt.show()

    ### training and prediction methods:
    @property
    def nn_model(self):
        """ Needs to be overwritten with reference to actual model (e.g. lstm_model, transformer_model, ...). """
        raise NotImplementedError("This property needs to be overwritten in subclasses with reference to actual model (e.g. lstm_model, transformer_model, ...).")

    def save_model_file(self):
        """ Needs to be overwritten with specific model mechanics. """
        raise NotImplementedError("This method needs to be overwritten in subclasses with specific model mechanics.")

    def run_training(self, custom_n_epochs: int = None, custom_early_stopping_patience: int = None,
                     visualise_validation_predictions_every: int = None):
        """ Train transformer model. """
        if custom_n_epochs is not None: self._n_train_epochs = custom_n_epochs  # don't set property here but attribute because otherwise run_training is re-triggered
        if custom_early_stopping_patience is not None: self._early_stopping_patience = custom_early_stopping_patience

        # initialise optimiser and scheduler:
        optimiser = optim.Adam(self.nn_model.parameters(), lr=self.initial_lr, betas=(0.9, 0.98),
                               eps=1e-9)  # beta and eps are standard values derived from github
        if self.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimiser, self.step_scheduler_step_size, gamma=0.1)
        elif self.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=self.plateau_scheduler_factor,
                                                             patience=5)
        else:
            raise ValueError("scheduler_to_use has to be either 'step' or 'plateau'!")

        # training loop:
        progress_bar = tqdm(range(self.n_train_epochs),
                            desc=f'Train loss: - | Val Loss: - | Patience {'/' if self.early_stopping_patience == 0 else f'{0}/{self.early_stopping_patience}'} | LRate: - | Progress')
        loss_train_history, loss_val_history = [], []
        for epoch in progress_bar:
            # eventually visualise training progress:
            if visualise_validation_predictions_every is not None:
                if epoch % visualise_validation_predictions_every == 0:
                    print("Plotting validation split prediction overview.")
                    self.plot_prediction_overview(custom_plot_title=f"Validation Split Predictions Epoch {epoch}")

            # conduct training step:
            loss_train, lr_train = self.nn_model.run_epoch(self.dataloader_train, optimiser=optimiser,
                                                             device=self.device, loss_criterion=self.loss_criterion,
                                                             is_training=True)
            loss_val, _ = self.nn_model.run_epoch(self.dataloader_val, optimiser=optimiser, device=self.device,
                                                    loss_criterion=self.loss_criterion, is_training=False)

            # every training epoch needs to reset recent predictions:
            self._predictions_val = self._predictions_train = None

            # scheduler step:
            if self.lr_scheduler == 'plateau':
                scheduler.step(loss_val)
            else:
                scheduler.step()

            # eventually randomise validation and training data:
            if self.randomise_validation:
                self.split_data(verbose=False)  # verbose=False to prevent status messages in every epoch

            loss_train_history.append(loss_train);
            loss_val_history.append(loss_val)

            # early stopping check:
            if self.early_stopping_patience != 0:
                # initialisation of vars in epoch 0:
                if epoch == 0:
                    best_loss = loss_val
                    counter = 0
                    continue
                # validation loss improved:
                if loss_val < best_loss:
                    best_loss = loss_val
                    counter = 0
                else:  # validation loss didn't improve
                    counter += 1
                    if counter >= self.early_stopping_patience:
                        print("Early stopping triggered at validation loss of", loss_val)
                        self._n_train_epochs = epoch + 1
                        break

            # progress bar for visualisation:
            progress_bar.desc = f'Train loss: {loss_train} | Val Loss: {loss_val}  | Patience {'/' if self.early_stopping_patience == 0 else f'{counter}/{self.early_stopping_patience}'} | LRate: {lr_train} | Progress'

        # save final losses:
        self._loss_train = loss_train
        self._loss_val = loss_val

        # save model:
        if self.model_save_directory is not None:
            self.save_model_file(
                custom_title_identifier=f"TrainL{loss_train} ValL{loss_val} TrainHR{self.hit_rate_train} ValHR{self.hit_rate_val}" if self.evaluate_hit_rate_upon_training else f"TrainL{loss_train} ValL{loss_val}")

        # loss progression plot and training info:
        if self.verbose:
            print(
                f"Training finished.\nFinal loss training data: {loss_train}\t\t\t\tValidation data: {loss_val}", f"\nFinal hit-rate training data: {self.hit_rate_train}\t\t\tValidation data: {self.hit_rate_val}" if self.evaluate_hit_rate_upon_training else "")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(self.n_train_epochs), loss_train_history, label='Training loss', color='blue')
            ax.plot(range(self.n_train_epochs), loss_val_history, label='Validation loss', color='red')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.grid(True)
            ax.legend()
            ax.set_title('Training Progress')

    def predict(self, input_values: Union[pd.Series, np.array],
                input_dates: np.array = None,
                dtype: Literal['pandas', 'numpy'] = 'numpy',
                return_tendency: bool = False,):
        """ Predict prices on new values. """
        try:
            input_dates = np.array(input_values.index, dtype=np.datetime64) if input_dates is None else input_dates
            dates_provided = True
        except AttributeError:
            dates_provided = False
        input_values = np.array(input_values, dtype=np.float32)

        # normalise input:
        normalised_input = np.array(self.normaliser.transform(input_values), dtype=np.float64)

        # convert to required shape (batch_size, sequence_length, features)
        input_tensor = torch.unsqueeze(torch.Tensor(normalised_input), dim=0)
        input_tensor = torch.unsqueeze(input_tensor, dim=2)
        if input_tensor.size()[1] != self.rolling_window_size:
            raise ValueError(f"Input values length needs to match model's rolling window size ({self.rolling_window_size})")

        # call model and re-transform input:
        predictions = self.nn_model(input_tensor)
        predictions = predictions.cpu().detach().numpy()
        predictions = np.squeeze(self._normaliser.inverse_transform(predictions))
        if dates_provided:
            prediction_dates = pd.date_range(input_dates.max() + pd.Timedelta(f'{self.sampling_rate_minutes}min'),
                                             input_dates.max() + self.forecast_horizon * pd.Timedelta(
                                                 f'{self.sampling_rate_minutes}min'),
                                             freq=f"{self.sampling_rate_minutes}min")
        else:
            prediction_dates = range(len(input_values), len(input_values) + len(predictions))
        tendency = 'up' if (predictions[-1] > input_values[-1]) else 'down'

        if self.verbose:
            if dates_provided: input_dates = pd.to_datetime(input_dates)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(input_dates if dates_provided else range(len(input_values)), input_values, color='blue', label='Input Prices')
            ax.plot(prediction_dates,
                    predictions, color='green', label='Predicted Prices')
            print(f'Prices are expected to go {tendency.upper()}!')
            # formatting:
            ax.set_xlabel('Date' if dates_provided else 'Timestep')
            ax.set_ylabel('Price')
            ax.set_title('Prediction Overview')
            ax.legend()

        # what to return:
        if dtype == 'pandas': output = pd.Series(index=prediction_dates, data=predictions)
        else: output = predictions, prediction_dates
        if return_tendency: return output + tuple([tendency])
        else: return output


class LSTMPredictor(NNPredictor):
    """ LSTM based stock price predictor framework. """
    def __init__(self,
                 # base class parameters:
                 sampling_rate_minutes: int = 15,  # data import parameters
                 price_csv_path: str = None,
                 price_column: str = 'close',
                 date_column: str = 'date',
                 daily_prediction_hour: int = None,  # data preparation parameters
                 predict_before_daily_prediction_hour: bool = False,
                 rolling_window_size: int = 32,
                 forecast_horizon: int = 12,
                 validation_split: float = 0.2,
                 randomise_validation: bool = True,
                 batch_size: int = 32,
                 forecast_step_loss_weight_range: [float, float] = (1, 0.7),  # eval parameter
                 use_mps_if_available: bool = False,  # training parameter

                 # lstm-specific parameters:
                 model_load_file_path: str = None,  # model parameters
                 hidden_lstm_layer_size: int = 64,
                 n_lstm_layers: int = 3,
                 dropout: float = 0.3,
                 use_final_hidden_state: bool = False,
                 use_pre_lstm_fc_layer: bool = True,
                 init_weights: bool = True,

                 model_save_directory: str = None,
                 n_train_epochs: int = None,  # if set to some number, trains upon initialisation
                 lr_scheduler: Literal['step', 'plateau'] = 'plateau',
                 initial_lr: float = 0.001,
                 step_scheduler_step_size: int = 40,
                 plateau_scheduler_factor: float = 0.5,
                 early_stopping_patience: int = 30,  # if 0 doesn't utilise early stopping
                 verbose=True,  # other parameters
                 evaluate_hit_rate_upon_training=True,  # can save run-time especially for short training procedures
                 ):
        super().__init__(price_csv_path=price_csv_path, date_column=date_column,
                         price_column=price_column, rolling_window_size=rolling_window_size,
                         forecast_horizon=forecast_horizon, sampling_rate_minutes=sampling_rate_minutes,
                         validation_split=validation_split, randomise_validation=randomise_validation,
                         daily_prediction_hour=daily_prediction_hour,
                         predict_before_daily_prediction_hour=predict_before_daily_prediction_hour,
                         batch_size=batch_size, forecast_step_loss_weight_range=forecast_step_loss_weight_range,
                         use_mps_if_available=use_mps_if_available,
                         model_save_directory=model_save_directory,
                         n_train_epochs=n_train_epochs,
                         lr_scheduler=lr_scheduler,
                         initial_lr=initial_lr,
                         step_scheduler_step_size=step_scheduler_step_size,
                         plateau_scheduler_factor=plateau_scheduler_factor,
                         early_stopping_patience=early_stopping_patience,
                         evaluate_hit_rate_upon_training=evaluate_hit_rate_upon_training,
                         verbose=verbose)
        self.verbose = verbose

        ### provided model parameters:
        self._hidden_lstm_layer_size = hidden_lstm_layer_size
        self._n_lstm_layers = n_lstm_layers
        self._dropout = dropout
        self._use_final_hidden_state = use_final_hidden_state
        self._use_pre_lstm_fc_layer = use_pre_lstm_fc_layer
        self._init_weights = init_weights

        # model placeholder and import:
        self._lstm_model = None
        if model_load_file_path is not None: self.load_model_from_pt_file(model_load_file_path)

        # eventually start training:
        if n_train_epochs is None:
            if verbose:
                print(
                    'No training epochs defined upon initialisation. Define LSTMPredictor.n_train_epochs to start training procedure.')
        else:
            self.run_training()
            if verbose:
                print('Training finished. Plotting results for validation split:')
                self.plot_prediction_overview(day_slice=(0, 5))

    # str and print operators:
    def __str__(self):
        return self.describe()

    def __repr__(self):
        return self.describe()

    def describe(self):
        intro_str = "------------------- LSTMPredictor Instance -------------------\n\n"
        data_str = f"Data Attributes:\n- sampling rate: {self.sampling_rate_minutes} min (= {self.sampling_rate_minutes / 60} h = {self.sampling_rate_minutes /60 /14} d)\n- rolling window size: {self._rolling_window_size}\n- forecast horizon: {self._forecast_horizon}\n- daily prediction hour: {f'{self._daily_prediction_hour}:00\n- predicting at last observation before prediction hour: {self.predict_before_daily_prediction_hour}' if self.daily_prediction_hour is not None else 'None'}\n- validation split: {self._validation_split}\n- randomise validation: {self.randomise_validation}\n- amount of training observations: {len(self.X_train)}\n- amount of validation observations: {len(self.X_val)}\n\n"
        model_str = f"Model Attributes:\n- hidden LSTM layers: {self.hidden_lstm_layer_size}\n- number of LSTM layers: {self.n_lstm_layers}\n- pre LSTM fully connected layer: {self.use_pre_lstm_fc_layer}\n\n"
        if self.n_train_epochs is not None:
            training_str = f"Training Attributes:\n- final training loss: {self.loss_train}\n- final validation loss: {self.loss_val}\n- final training hit-rate: {self.hit_rate_train}\n- final validation hit-rate: {self.hit_rate_val}"
        else:
            training_str = "Model wasn't trained yet (or was imported)."
        return intro_str + data_str + model_str + training_str

    # model parameters:
    @property
    def hidden_lstm_layer_size(self):
        return self._hidden_lstm_layer_size

    @hidden_lstm_layer_size.setter
    def hidden_lstm_layer_size(self, value):
        """ Changing value re-initialises LSTM model. """
        self._hidden_lstm_layer_size = value
        self._lstm_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def n_lstm_layers(self):
        return self._n_lstm_layers

    @n_lstm_layers.setter
    def n_lstm_layers(self, value):
        """ Changing value re-initialises LSTM model. """
        self._n_lstm_layers = value
        self._lstm_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, value):
        """ Changing value re-initialises LSTM model. """
        self._dropout = value
        self._lstm_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def use_final_hidden_state(self):
        return self._use_final_hidden_state

    @use_final_hidden_state.setter
    def use_final_hidden_state(self, value):
        """ Changing value re-initialises LSTM model. """
        self._use_final_hidden_state = value
        self._lstm_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def use_pre_lstm_fc_layer(self):
        return self._use_pre_lstm_fc_layer

    @use_pre_lstm_fc_layer.setter
    def use_pre_lstm_fc_layer(self, value):
        """ Changing value re-initialises LSTM model. """
        self._use_pre_lstm_fc_layer = value
        self._lstm_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def init_weights(self):
        return self._init_weights

    @init_weights.setter
    def init_weights(self, value):
        """ Changing value re-initialises LSTM model. """
        self._init_weights = value
        self._lstm_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def lstm_model(self):
        if self._lstm_model is None:
            self._lstm_model = LSTMModel(input_size=1, hidden_layer_size=self.hidden_lstm_layer_size,
                                         num_layers=self._n_lstm_layers, n_forecast_steps=self.forecast_horizon,
                                         dropout=self.dropout, use_pre_lstm_fc_layer=self.use_pre_lstm_fc_layer,
                                         use_final_hidden_state=self.use_final_hidden_state,
                                         use_hidden_states_across_forecast_steps=True,
                                         init_weights=self.init_weights)
            self._lstm_model.to(self.device)
            try:
                self._lstm_model = torch.compile(self._lstm_model)
            except RuntimeError as e:
                if self.verbose:
                    print(e)
                    print("Skipping compilation. This will not influence model operation.")
        return self._lstm_model

    @lstm_model.setter
    def lstm_model(self, value):
        self._lstm_model = value
        # set other properties based on model parameters:
        try:
            a = value.linear_1  # try accessing layer
            self._use_pre_lstm_fc_layer = True if (a is not None) else False
        except AttributeError:  # if not found
            self._use_pre_lstm_fc_layer = False
        self._hidden_lstm_layer_size = value.lstm.hidden_size
        self._n_lstm_layers = value.lstm.num_layers
        self._dropout = value.lstm.dropout
        self._predictions_val = self._predictions_train = None

    # important to allow methods of base-class to work:
    @property
    def nn_model(self):
        """ Link subclass model (LSTM) to base class. """
        return self.lstm_model

    def load_model_from_pt_file(self, file_path: str):
        """ Load LSTM model from .pt file. """
        self.lstm_model = torch.load(file_path, weights_only=False)

    def save_model_file(self, custom_save_directory: str = None, custom_title_identifier: str = None):
        """ Save model to LSTMPredictor.model_save_directory or provided custom_save_directory. """
        assert custom_save_directory is not None or self.model_save_directory is not None;
        "Either custom_save_directory needs to be passed to function or LSTMPredictor.model_save_directory needs to be defined!"
        save_path = custom_save_directory if custom_save_directory is not None else self.model_save_directory
        save_title = filemgmt.file_title(
            f"LSTM Model RW{self.rolling_window_size} FH{self.forecast_horizon} Layers{self.n_lstm_layers} Size{self.hidden_lstm_layer_size}{f' {custom_title_identifier}' if custom_title_identifier is not None else ''}",
            dtype_suffix=".pt")
        if self.verbose: print(f"Saving LSTM model to {save_path}/{save_title}.pt")
        torch.save(self.lstm_model, save_path / save_title)


class TransformerPredictor(NNPredictor):
    """ Transformer-based stock price predictor framework. """
    def __init__(self,
                 # base class parameters:
                 sampling_rate_minutes: int = 15,  # data import parameters
                 price_csv_path: str = None,
                 price_column: str = 'close',
                 date_column: str = 'date',
                 daily_prediction_hour: int = None,  # data preparation parameters
                 predict_before_daily_prediction_hour: bool = False,
                 rolling_window_size: int = 32,
                 forecast_horizon: int = 12,
                 validation_split: float = 0.2,
                 randomise_validation: bool = True,
                 batch_size: int = 32,
                 forecast_step_loss_weight_range: [float, float] = (1, 0.7),  # eval parameter
                 use_mps_if_available: bool = False,  # training parameter

                 # transformer-specific parameters:
                 model_load_file_path: str = None,
                 hidden_transformer_layer_size: int = 64,
                 n_transformer_layers: int = 3,
                 n_transformer_heads: int = 4,
                 dropout: float = 0.3,
                 use_pre_transformer_fc_layer: bool = True,
                 init_weights: bool = True,

                 # training parameters:
                model_save_directory: str = None,
                 n_train_epochs: int = None,  # if set to some number, trains upon initialisation
                 lr_scheduler: Literal['step', 'plateau'] = 'plateau',
                 initial_lr: float = 0.001,
                 step_scheduler_step_size: int = 40,
                 plateau_scheduler_factor: float = 0.5,
                 early_stopping_patience: int = 30,  # if 0 doesn't utilise early stopping
                 verbose=True,  # other parameters
                 evaluate_hit_rate_upon_training=True,  # can save run-time especially for short trainig procedures
                 ):
        super().__init__(price_csv_path=price_csv_path, date_column=date_column,
                         price_column=price_column, rolling_window_size=rolling_window_size,
                         forecast_horizon=forecast_horizon, sampling_rate_minutes=sampling_rate_minutes,
                         validation_split=validation_split, randomise_validation=randomise_validation,
                         daily_prediction_hour=daily_prediction_hour,
                         predict_before_daily_prediction_hour=predict_before_daily_prediction_hour,
                         batch_size=batch_size, forecast_step_loss_weight_range=forecast_step_loss_weight_range,
                         use_mps_if_available=use_mps_if_available,
                         model_save_directory=model_save_directory,
                         n_train_epochs=n_train_epochs,
                         lr_scheduler=lr_scheduler,
                         initial_lr=initial_lr,
                         step_scheduler_step_size=step_scheduler_step_size,
                         plateau_scheduler_factor=plateau_scheduler_factor,
                         early_stopping_patience=early_stopping_patience,
                         evaluate_hit_rate_upon_training=evaluate_hit_rate_upon_training,
                         verbose=verbose)
        self.verbose = verbose

        ### provided model parameters:
        self._hidden_transformer_layer_size = hidden_transformer_layer_size
        self._n_transformer_layers = n_transformer_layers
        self._n_transformer_heads = n_transformer_heads
        self._dropout = dropout
        self._use_pre_transformer_fc_layer = use_pre_transformer_fc_layer
        self._init_weights = init_weights

        # model placeholder and import:
        self._transformer_model = None
        if model_load_file_path is not None: self.load_model_from_pt_file(model_load_file_path)

        # eventually start training:
        if n_train_epochs is None:
            if verbose:
                print(
                    'No training epochs defined upon initialisation. Define n_train_epochs to start training procedure.')
        else:
            self.run_training()
            if verbose:
                print('Training finished. Plotting results for validation split:')
                self.plot_prediction_overview(day_slice=(0, 5))

    # str and print operators:
    def __str__(self):
        return self.describe()

    def __repr__(self):
        return self.describe()

    def describe(self):
        intro_str = "------------------- TransformerPredictor Instance -------------------\n\n"
        data_str = f"Data Attributes:\n- sampling rate: {self.sampling_rate_minutes} min (= {self.sampling_rate_minutes / 60} h = {self.sampling_rate_minutes /60 /14} d)\n- rolling window size: {self._rolling_window_size}\n- forecast horizon: {self._forecast_horizon}\n- daily prediction hour: {f'{self._daily_prediction_hour}:00\n- predicting at last observation before prediction hour: {self.predict_before_daily_prediction_hour}' if self.daily_prediction_hour is not None else 'None'}\n- validation split: {self._validation_split}\n- randomise validation: {self.randomise_validation}\n- amount of training observations: {len(self.X_train)}\n- amount of validation observations: {len(self.X_val)}\n\n"
        model_str = f"Model Attributes:\n- hidden transformer layer size: {self.hidden_transformer_layer_size}\n- number of transformer layers: {self.n_transformer_layers}\n- number of transformer heads: {self.n_transformer_heads}\n- pre transformer fully connected layer: {self.use_pre_transformer_fc_layer}\n\n"
        if self.n_train_epochs is not None:
            training_str = f"Training Attributes:\n- final training loss: {self.loss_train}\n- final validation loss: {self.loss_val}\n- final training hit-rate: {self.hit_rate_train}\n- final validation hit-rate: {self.hit_rate_val}"
        else:
            training_str = "Model wasn't trained yet (or was imported)."
        return intro_str + data_str + model_str + training_str

    # model parameters:
    @property
    def hidden_transformer_layer_size(self):
        return self._hidden_transformer_layer_size

    @hidden_transformer_layer_size.setter
    def hidden_transformer_layer_size(self, value):
        """ Changing value re-initialises transformer model. """
        self._hidden_transformer_layer_size = value
        self._transformer_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def n_transformer_layers(self):
        return self._n_transformer_layers

    @n_transformer_layers.setter
    def n_transformer_layers(self, value):
        """ Changing value re-initialises transformer model. """
        self._n_transformer_layers = value
        self._transformer_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, value):
        """ Changing value re-initialises transformer model. """
        self._dropout = value
        self._transformer_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def n_transformer_heads(self):
        return self._n_transformer_heads

    @n_transformer_heads.setter
    def n_transformer_heads(self, value):
        """ Changing value re-initialises transformer model. """
        self._n_transformer_heads = value
        self._transformer_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def use_pre_transformer_fc_layer(self):
        return self._use_pre_transformer_fc_layer

    @use_pre_transformer_fc_layer.setter
    def use_pre_transformer_fc_layer(self, value):
        """ Changing value re-initialises transformer model. """
        self._use_pre_transformer_fc_layer = value
        self._transformer = None
        self._predictions_val = self._predictions_train = None

    @property
    def init_weights(self):
        return self._init_weights

    @init_weights.setter
    def init_weights(self, value):
        """ Changing value re-initialises transformer model. """
        self._init_weights = value
        self._transformer_model = None
        self._predictions_val = self._predictions_train = None

    @property
    def transformer_model(self) -> TransformerModel:
        if self._transformer_model is None:
            self._transformer_model = TransformerModel(input_size=1,
                                                       hidden_layer_size=self.hidden_transformer_layer_size,
                                                       num_layers=self.n_transformer_layers,
                                                       n_heads=self.n_transformer_heads,
                                                       n_forecast_steps=self.forecast_horizon,
                                                       dropout=self.dropout,
                                                       use_pre_transformer_fc_layer=self.use_pre_transformer_fc_layer,
                                                       init_weights=self.init_weights,
                                                       max_seq_len=self.rolling_window_size,)
            self._transformer_model.to(self.device)
            try:
                self._transformer_model = torch.compile(self._transformer_model)
            except RuntimeError as e:
                if self.verbose:
                    print(e)
                    print("Skipping compilation. This will not influence model operation.")
        return self._transformer_model

    # todo: check property names!
    @transformer_model.setter
    def transformer_model(self, value: TransformerModel):
        self._transformer_model = value
        # set other properties based on model parameters:
        try:
            a = value.linear_1  # try accessing layer
            self._use_pre_transformer_fc_layer = True if (a is not None) else False
        except AttributeError:  # if not found
            self._use_pre_transformer_fc_layer = False
        self._n_transformer_heads = value.transformer_encoder.encoder_layer.nhead
        self._hidden_transformer_layer_size = value.transformer_encoder.encoder_layer.dim_feedforward
        self._n_transformer_layers = value.transformer_encoder.num_layers
        self._dropout = value.transformer_encoder.encoder_layer.dropout

        # reset predictions:
        self._predictions_val = self._predictions_train = None

    # important to allow methods of base-class to work:
    @property
    def nn_model(self):
        """ Link subclass model (transformer) to base class. """
        return self.transformer_model

    def load_model_from_pt_file(self, file_path: str):
        """ Load transformer model from .pt file. """
        self.transformer_model = torch.load(file_path, weights_only=False)

    def save_model_file(self, custom_save_directory: str = None, custom_title_identifier: str = None):
        """ Save model to TransformerPredictor.model_save_directory or provided custom_save_directory. """
        assert custom_save_directory is not None or self.model_save_directory is not None;
        "Either custom_save_directory needs to be passed to function or LSTMPredictor.model_save_directory needs to be defined!"
        save_path = custom_save_directory if custom_save_directory is not None else self.model_save_directory
        save_title = filemgmt.file_title(
            f"Transformer Model RW{self.rolling_window_size} FH{self.forecast_horizon} Layers{self.n_transformer_layers} Size{self.hidden_transformer_layer_size} Heads{self.n_transformer_heads}{f' {custom_title_identifier}' if custom_title_identifier is not None else ''}",
            dtype_suffix=".pt")
        if self.verbose: print(f"Saving Transformer model to {save_path}/{save_title}.pt")
        torch.save(self.transformer_model, save_path / save_title)