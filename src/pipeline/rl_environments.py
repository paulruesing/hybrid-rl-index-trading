from statistics import median

import src.pipeline.preprocessing as preprocessing
from src.pipeline.predictors import LSTMPredictor
from src.pipeline.financial_products import KOCertificate, KOCertificateSet
from src.utils import file_management as filemgmt
from src.pipeline.rl_agents import MultiProductAgent

from itertools import product
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from typing import Union, Literal
import numpy as np
import pandas as pd
import enum
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


class RLTradingEnv(gym.Env):
    """
    A reinforcement learning environment for trading using structured price data and predictive signals.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 price_csv_path: Union[str, Path],
                 price_sampling_rate_minutes: int = 1,
                 price_column='close',
                 date_column='date',
                 predictor_instances: [LSTMPredictor] = None,  # to be imported for data preparation
                 precalculate_predictor_observations: bool = True,
                 product_set: KOCertificateSet = None,
                 leverage_categories: [float] = (1.0, 2.0, 3.0, 4.0, 5.0),  # used to create action space
                 include_open_leverage_category: bool = False,  # if True, larger than last entry is highest category
                 trading_quantity_per_leverage_factor: float = 1.0,
                 # at 1.0 each trade is sized acccording to total balance / amount of leverage categories
                 possible_trade_quantities: Literal['all', 'fixed', 'arbitrary'] = 'all',  # used to define action space
                 sell_opposite_direction_if_no_cash: bool = True,
                 sell_all_opposite_products_if_no_cash: bool = True,
                 starting_cash=1000000,
                 commission_rate=0.001,  # reflects a typical spread, because we can trade commission-free through wikifolio
                 verbose=True,
                 potential_horizon_days: int = 90,
                 ):
        """
        A reinforcement learning environment for trading using structured price data and predictive signals.

        This environment wraps a trading simulation where an agent interacts with financial instruments
        (e.g., knockout certificates) and receives predictions from one or more LSTM-based predictors.
        Actions are taken based on leverage categories, and rewards are based on changes in portfolio value.

        The environment steps once per day for each daily prediction hour defined by the predictors.

        Parameters
        ----------
        price_csv_path : str or Path
            Path to CSV file containing historical price data.
        price_sampling_rate_minutes : int, default 1
            Frequency of the price data in minutes.
        price_column : str, default 'close'
            Column name in the CSV representing the price.
        date_column : str, default 'date'
            Column name in the CSV representing the timestamp.
        predictor_instances : list of LSTMPredictor
            Predictors used to generate potential signals.
        precalculate_predictor_observations : bool, default True
            If true, calculates all potential estimates upon accessing first observation (run-time efficient for training).
        product_set : KOCertificateSet
            Set of financial products (e.g., long/short leverage instruments).
        leverage_categories : list of float, default (1.0, 2.0, 3.0, 4.0, 5.0)
            Leverage brackets used to define the action space.
        include_open_leverage_category : bool, default False
            Whether to include an additional category for leverage greater than the last entry.
        trading_quantity_per_leverage_factor : float, default 1.0
            Determines trade size as a fraction of balance per leverage category.
        sell_opposite_direction_if_no_cash : bool, default True
            If insufficient cash for buying operation: sell opposite certificates.
        sell_all_opposite_products_if_no_cash : bool, default True
            If sell_opposite_products_if_no_cash is True and this is True, sells ALL opposite products.
            If this is False, sells only opposite leverage (if long = 1, short = 5) of products.
        starting_cash : float, default 1000000
            Initial cash balance for the agent.
        commission_rate : float, default 0.001
            Effective commission rate or spread applied on each transaction.
        verbose : bool, default True
            Whether to print detailed step and trade information.
        potential_horizon_days : int, default is 90
            Forecast days to scale expected return potential to. Only relevant for info statements.
        """
        super(RLTradingEnv, self).__init__()  # initialise base class

        self._price_csv_path = price_csv_path
        self._price_column = price_column
        self._date_column = date_column
        self._price_series = None  # placeholder for property
        self.price_sampling_rate_minutes = price_sampling_rate_minutes

        self.starting_cash = starting_cash
        self.commission_rate = commission_rate
        self.verbose = verbose
        self.potential_horizon_days = potential_horizon_days

        # current episode (sequence of steps)
        self.current_episode = 0

        # read in predictors:
        self.predictor_instances = predictor_instances
        for predictor in self.predictor_instances: predictor.verbose = False  # mute predictors to prevent unexpected console output
        self._daily_prediction_hours = None  # important predictor setting for sampling

        # this influences the action space:
        self._product_set = product_set
        self._leverage_categories = leverage_categories
        self._include_open_leverage_category = include_open_leverage_category
        self.trading_quantity_per_leverage_factor = trading_quantity_per_leverage_factor
        self.sell_opposite_direction_if_no_cash = sell_opposite_direction_if_no_cash
        self.sell_all_opposite_products_if_no_cash = sell_all_opposite_products_if_no_cash
        self._action_enum = None

        self.action_space = spaces.Discrete(len(self.action_enum))  # is re-written within self.action_enum property

        # step dates (property to be calculated upon read-out)
        self._step_dates_list = self._step_timestamp_list = None
        self.current_step = 0;
        self.init_start_step()  # sets self.current_step to minimum value for predictors to be callable

        # set of floats representing available information for agent:
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(len(predictor_instances) + 2,),
                                            dtype=np.float64)
        self._potential_estimates_dict = None
        self.precalculate_predictor_observations = precalculate_predictor_observations
        # contains floats of each predictor's output and two more for current cash and holding


        # possible range of rewards for actions:
        self.reward_range = (-np.inf, np.inf)

        # initial conditions:
        self.cash = self.starting_cash
        self.shares_per_product = pd.Series(index=self.product_set.by_isin.keys(), name='Shares')
        self.shares_per_product.iloc[:] = 0.0

        # todo: remove
        self.shares = 0

    def step(self, action: int, track_portfolio_exposure: bool = True):
        """
        Run one timestep of the environment’s dynamics.

        This method executes the given action, updates the internal state, calculates the reward,
        and returns the next observation, the reward, a boolean indicating if the episode is done,
        and a diagnostic info dictionary.

        Parameters
        ----------
        action : int
            Action index corresponding to the current action enumeration.
        track_portfolio_exposure : bool, default True
            Whether to track portfolio exposure during each step. Run-time costly.

        Returns
        -------
        observation : np.ndarray
            Current state observation including predictor signals, cash, and holdings.
        reward : float
            Reward from the action taken, equal to change in portfolio value.
        done : bool
            Whether the current episode is finished.
        truncated : bool
            Whether the current episode was truncated.
        info : dict
            Additional diagnostic information for debugging and logging.
        """
        # infer current balance:
        balance = self.current_balance

        # increase step
        self.current_step += 1
        if self.current_step == self.total_steps:  # if episode is finished
            done = True
            self.current_episode = self.next_episode()
            self.init_start_step()
        else:
            done = False

        # take action:
        self.take_action(action)

        # compute status and calculate reward:
        reward = self.current_balance - balance  # equals change of balance
        # done = (self.current_step == self.total_steps - 1)  # see if episode is finished

        # compute average current exposure:
        if track_portfolio_exposure:
            portfolio_exposure = np.sum(self.open_positions['Leverage']
                                        * self.open_positions['% Portfolio'] / 100  # weight by share of portfolio
                                        * np.where(self.open_positions['Direction'] == 'long',
                                                   1.0, -1.0)  # multiply with 1.0 or -1.0 depending on direction of product
                                        )
            portfolio_exposure = portfolio_exposure.item() if (portfolio_exposure is not np.nan) and (
                        portfolio_exposure != 0) else 0.0
        else: portfolio_exposure = None

        # get current observation:
        obs = self.current_observation
        # todo: if tendencies included in observations, include such here

        # construct info dictionary:
        info = {'Step': self.current_step,
                'Time': self.current_step_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Reward': round(reward.item(), 2) if (reward is not np.nan) and (reward != 0) else 0,
                'Action': self.action_enum_dict[action],
                f'Avg. Expected Potential / {self.potential_horizon_days}d': self.current_avg_scaled_predicted_potential,
                'Cash': round(self.cash, 2).item() if isinstance(round(self.cash, 2), np.float64) else round(self.cash,
                                                                                                             2),
                'Total': round(self.current_balance, 2).item() if isinstance(round(self.current_balance, 2),
                                                                             np.float64) else round(
                    self.current_balance, 2)}

        if portfolio_exposure is not None:
            info['Total Exposure'] = portfolio_exposure

        # if done: self.reset()  # happens automatically!
        # current observation property constructs observation space
        return obs, reward, done, False, info  # gymnasium returns terminated, truncated check

    def take_action(self, action: int):
        """
        Execute the given action and update environment state accordingly.

        Handles both buy and sell logic, including selection of products from the product set,
        calculation of investable amount, share buying/selling, and updating cash and holdings.

        Parameters
        ----------
        action : int
            Action index corresponding to the current action enumeration.
        """
        # classify action according to -> type, direction, leverage_span
        if action == 0:
            return
        type, direction, low_leverage, _, high_leverage = self.action_enum_dict[action].split()
        leverage_span = (float(low_leverage.replace('x', '')), float(high_leverage.replace('x', '')))

        # buy action:
        if type == 'Buy':
            try:  # find respective product and safe isin
                isin = self.product_set.get_product_from_leverage_span(date=self.current_step_timestamp,
                                                                       direction=direction, leverage_span=leverage_span)
            except KeyError:  # no product found
                if self.verbose: print(f"No product with leverage inside {leverage_span} found. Using next smaller leverage.")
                try:
                    isin = self.product_set.get_product_from_leverage_span(date=self.current_step_timestamp,
                                                                           direction=direction,
                                                                           leverage_span=(1.0, leverage_span[1]),
                                                                           # include all smaller leverages
                                                                           search_ascending=False,
                                                                           # sort in decreasing order
                                                                           )
                except KeyError:  # still no product found, then return
                    if self.verbose: print(f"No product with smaller leverage available. Return")
                    return

            # current price:
            price = self.current_prices_per_product[isin] * (1 + self.commission_rate)

            # investable amount (fixed ratio per direction and leverage category of total portfolio balance):
            shares_to_buy = np.floor(
                self.current_balance / len(self.leverage_categories) * self.trading_quantity_per_leverage_factor
                / price)
            maximum_buyable_shares = np.floor(self.cash / price)  # cannot buy more shares than cash
            shares_to_buy = np.min([maximum_buyable_shares, shares_to_buy])

            # conduct purchase:
            self.shares_per_product[isin] += shares_to_buy
            self.cash -= shares_to_buy * price

            if self.verbose:
                print(
                    f"[STEP {self.current_step}] Bought {shares_to_buy} shares of {isin} ({direction} with leverage in {leverage_span}) at {price}.")
                print(f"    Cash: {self.cash}, Holding: {self.current_holding}")

            # todo: think, whether here a threshold is better than == 0
            if maximum_buyable_shares == 0 and self.sell_opposite_direction_if_no_cash:
                opposite_direction = 'short' if direction == 'long' else 'long'

                # sell all opposite products:
                if self.sell_all_opposite_products_if_no_cash:
                    opposite_leverage = 1.0
                else:  # derive opposite leverage:
                    opposite_leverage_index = len(self.leverage_categories) - self.leverage_categories.index(
                        leverage_span[0]) - 2
                    if opposite_leverage_index == -1: opposite_leverage_index = 0  # if buying highest leverage, e.g. (4.5-5) or (>5) sell all low positions
                    opposite_leverage = self.leverage_categories[opposite_leverage_index]

                if self.verbose: print(
                    f"Couldn't buy {direction} certificates because cash quote too low. Trying to sell {opposite_direction} certificates with leverage higher than {opposite_leverage} now.")

                # prepare opposite selling operation:
                type = 'Sell'
                leverage_span = (opposite_leverage, leverage_span[1])
                direction = opposite_direction

        if type == 'Sell':
            # select all products with leverages inside span and higher:
            # todo: reflect whether all products with higher leverages should remain, or whether only inside span should be sold
            open_candidates = self.open_positions.loc[self.open_positions.Direction == direction]
            leverage_candidates = open_candidates.loc[
                (open_candidates.Leverage >= leverage_span[0])]  # & open_candidates.Leverages <= leverage_span[1]]
            if len(leverage_candidates) == 0: return  # no product found

            # else fetch information:
            isins = list(leverage_candidates.index)
            prices = leverage_candidates.Price * (1 - self.commission_rate)
            shares_to_sell = leverage_candidates.Shares

            # conduct sale:
            self.shares_per_product[isins] = 0
            self.cash += (shares_to_sell * prices).sum()

            if self.verbose:
                print(
                    f"[STEP {self.current_step}] Sold\n{shares_to_sell}\nshares ({direction}s with leverage higher than {leverage_span[0]}) at\n{prices}.")
                print(f"    Cash: {self.cash}, Holding: {self.current_holding}")

    def next_episode(self):
        """
        Advance to the next episode index in a round-robin manner.

        Returns
        -------
        int
            The index of the next episode.
        """
        return (self.current_episode + 1) % self.total_episodes

    def reset(self, seed: int = None) -> (np.ndarray, dict):
        """
        Reset the environment to the initial state at the start of an episode.

        seed is placeholder currently for stable_baselines3

        Returns
        -------
        np.ndarray
            Initial observation after reset, including predictors' outputs, cash, and holdings.
        dict
            Info dict of reset action.
        """
        self.init_start_step()
        if self.verbose:  # info statement
            print(f"Starting episode {self.current_episode + 1}", (f"/ {self.total_episodes}"))
        self.cash = self.starting_cash
        self.shares_per_product.iloc[:] = 0
        return self.current_observation, {'Status': 'Starting new episode'}  # info dict

    def get_predictor_input(self, predictor: LSTMPredictor, custom_timestamp: pd.Timestamp = None) -> pd.Series:
        """
        Retrieve the time series input for a given predictor at the current step.

        Builds a rolling window view of the price series according to the predictor’s requirements.

        Parameters
        ----------
        predictor : LSTMPredictor
            The predictor instance whose input format is to be generated.
        custom_timestamp : int, optional
            Custom timestamp compute predictor input for. If None, uses self.current_step_timestamp

        Returns
        -------
        np.ndarray
            Sampled price window for prediction.
        """
        # infer properties:
        rolling_window_size = predictor.rolling_window_size
        sampling_rate_minutes = predictor.sampling_rate_minutes
        predict_before_daily_prediction_hour = predictor.predict_before_daily_prediction_hour

        # convert timestamp to datetime_int_index:
        timestamp = self.current_step_timestamp if custom_timestamp is None else custom_timestamp
        date_time_int_index = np.argwhere(self.price_series[:timestamp])[-1].item()  # last index = index of timestamp

        # calculate start int index for rolling window (if not predicting before daily prediction hour add 1 index:
        start_index = int(
            date_time_int_index - rolling_window_size * sampling_rate_minutes / self.price_sampling_rate_minutes)

        # sanity check:
        if start_index < 0: raise ValueError("Rolling window size too large for current step.")

        # slice according to start and end index (adjusted by +1 if prediction should be after prediction hour) and sampling rate:
        sampled_prices = self.price_series.iloc[
                         start_index + (not predict_before_daily_prediction_hour):date_time_int_index + (
                             not predict_before_daily_prediction_hour):int(
                             sampling_rate_minutes / self.price_sampling_rate_minutes)]

        return sampled_prices

    def init_start_step(self):
        """
        Initialize the current step to the earliest valid step for all predictors.

        Ensures that the rolling windows for all predictors can be computed.
        """
        self.current_step = 0
        step_too_small = True
        while step_too_small:
            try:
                # try to get predictor input:
                _ = [self.get_predictor_input(predictor) for predictor in self.predictor_instances]
                step_too_small = False  # if successful, step is sufficient
            except ValueError:  # if current step too small for rolling window view, increase by 1
                self.current_step += 1

        if self.verbose:
            print(
                f"Starting at {self.current_step_date} although data is provided since {self.price_series.iloc[0:1].index.item().isoformat()[:10]} to have sufficient information for all predictors.")

    ################ Action Space Properties ################
    @property
    def include_open_leverage_category(self):
        """
        bool : Whether action space contains an additional category for arbitrarily high leverage.
        """
        return self._include_open_leverage_category

    @property
    def product_set(self) -> KOCertificateSet:
        """"
        KOCertificateSet : Set of tradable instruments available to the agent.
        """
        return self._product_set

    @product_set.setter
    def product_set(self, value: KOCertificateSet):
        """ product_set setter. Resets action space. """
        # todo: rethink whether this should remain mutable
        self._action_enum = None
        self._product_set = value

    @property
    def leverage_categories(self) -> [float]:
        """
        list of float : List of leverage categories used to define actions.
        """
        return self._leverage_categories

    @leverage_categories.setter
    def leverage_categories(self, value: [float]):
        """ leverage_categories setter. Resets action space. """
        self._action_enum = None
        self._leverage_categories = value

    @property
    def action_enum(self) -> enum.Enum:
        """ Action enum class, dynamically created according to leverage_categories and product_set. """
        if self._action_enum is None:
            # create action space labels: order is Buy Long, Sell Long, Buy Short, Sell Short for each leverage category
            leverage_span_tuples = [
                (start, np.inf if ind + 1 == len(self.leverage_categories) else self.leverage_categories[ind + 1]) for
                ind, start in enumerate(self.leverage_categories)]
            if not self._include_open_leverage_category: leverage_span_tuples = leverage_span_tuples[
                                                                                :-1]  # remove last (open) leverage span
            labels = ['Hold'] + [f"{action} {dir} {start}x - {end}x" for (start, end), dir, action in
                                 product(leverage_span_tuples, ['long', 'short'], ['Buy', 'Sell'])]
            self._action_enum = enum.Enum('TradingActions', dict(zip(labels, range(len(labels)))))
            self.action_space = spaces.Discrete(len(self.action_enum))  # rewrite self.action_space
        return self._action_enum

    @property
    def action_enum_dict(self) -> {int: str}:
        """ Dict with enum values as keys and names as values. """
        return {entry.value: entry.name for entry in self.action_enum}

    ################ Training Properties ################
    @property
    def total_episodes(self):
        """ Total episodes equals amount of provided stock data frames. """
        # todo: method currently unnecessary
        return 1

    @property
    def total_steps(self):
        """ One step per day and distinct prediction hour. """
        return len(self.step_dates_list)

    @property
    def no_of_days(self):
        """ Number of days in the provided price series. """
        return int(len(self.step_dates_list) / len(
            self.daily_prediction_hours))  # step dates list has multiple entries for multiple prediction hours

    @property
    def daily_prediction_hours(self):
        if self._daily_prediction_hours is None:
            self._daily_prediction_hours = {predictor.daily_prediction_hour for predictor in
                                            self.predictor_instances}  # at every distinct prediction hour (sets correct for duplicates)
        return self._daily_prediction_hours

    @property
    def step_dates_list(self) -> [str]:
        """ Dates for each day in the price series at each distinct prediction hour. """
        if self._step_dates_list is None:
            grouped = self.price_series.groupby(
                by=[self.price_series.index.year, self.price_series.index.month, self.price_series.index.day]).min()

            # multiple entries for each distinct prediction hour:
            self._step_dates_list = [f"{year}-{month}-{day}" for (year, month, day), pred_hour in
                                     product(grouped.index, self.daily_prediction_hours)]

            # if last date doesn't have entries until highest daily_prediction_hour, remove last day from list:
            if self.price_series[self.step_dates_list[-1]].index.hour.max() < np.max(
                    list(self.daily_prediction_hours)).item():
                self._step_dates_list = self.step_dates_list[:-(len(self.daily_prediction_hours))]
        return self._step_dates_list

    @property
    def step_timestamp_list(self) -> [pd.Timestamp]:
        """ Timestamps for each day in the price series at each distinct prediction hour. Has nan values for spots without data. """
        if self._step_timestamp_list is None:
            temp_list = []
            # iterate over all pred_hour step_dates combinations
            for step_date, pred_hour in product(self.step_dates_list, self.daily_prediction_hours):
                time_str = f"{step_date} {f'0{pred_hour}' if pred_hour < 10 else pred_hour}"
                try:  # if no data is present for that timestamp-str
                    temp_list.append(self.price_series[time_str].iloc[0:1].index.item())
                except:  # nan for empty spots
                    temp_list.append(None)
            self._step_timestamp_list = temp_list
        return self._step_timestamp_list

    @property
    def current_prediction_hour(self) -> int:
        return list(self.daily_prediction_hours)[(self.current_step % len(self.daily_prediction_hours))]

    @property
    def current_step_date(self) -> str:
        return self.step_dates_list[self.current_step]

    @property
    def current_step_timestamp(self) -> pd.Timestamp:
        """ Return timestamp of current step's datetime. """
        candidate = self.step_timestamp_list[self.current_step]
        while candidate is None:  # timestamp list has None elements at timestamps where no data is present
            self.current_step += 1
            candidate = self.step_timestamp_list[self.current_step]
        return candidate

    ################ Observation Properties ################
    @property
    def current_avg_scaled_predicted_potential(self):
        """ Current average predicted potential for next potential_horizon_days (part of current_observation). """
        potential_list = np.array([])
        for type, horizon_minutes, potential in zip(self.observation_types, self.observation_horizons_minutes, self.current_observation):
            if type != 'potential': continue
            potential = potential / horizon_minutes * self.potential_horizon_days * 24 * 60  # # scale per minute and then to per self.potential_horizon_days days
            potential_list = np.append(potential_list, [potential])
        return np.nanmean(potential_list).item()

    @property
    def price_series(self):
        """ Underlying price series for observations and reward calculations. """
        if self._price_series is None:
            self._price_series = preprocessing.read_price_csv(csv_path=self.price_csv_path, date_column=self.date_column,
                                                                price_column=self.price_column)
        return self._price_series

    @property
    def price_csv_path(self):
        """ File path of csv file containing price series. """
        return self._price_csv_path

    @price_csv_path.setter
    def price_csv_path(self, value):
        """ Changing value re-downloads price series. """
        self._price_csv_path = value
        self._price_series = None

    @property
    def date_column(self) -> str:
        """ date column of csv file containing price series. """
        return self._date_column

    @date_column.setter
    def date_column(self, value):
        """ Changing value re-downloads price series. """
        self._date_column = value
        self._price_series = None

    @property
    def price_column(self) -> str:
        """ price column of csv file containing price series. """
        return self._price_column

    @price_column.setter
    def price_column(self, value):
        """ Changing value re-downloads price series. """
        self._price_column = value
        self._price_series = None

    @property
    def observation_horizons_minutes(self) -> [int]:
        """ The forecast horizon in minutes of each observation of type 'potential' or 'tendency'. Is None for other types. """
        predictor_horizons = [predictor.forecast_horizon * predictor.sampling_rate_minutes for predictor in self.predictor_instances]
        # tendency_horizons (placeholder!)
        other = [None, None]  # cash and holding
        return predictor_horizons + other

    @property
    def observation_types(self) -> [Literal['potential', 'tendency', 'cash', 'holding']]:
        """ The types of observations the agent expects in a fixed order, e.g. ['potential', 'cash', 'holding']."""
        predictor_types = ['potential' for predictor in self.predictor_instances]
        # tendency_types (placeholder!)
        other = ['cash', 'holding']
        return predictor_types + other

    def compute_predicted_potentials(self, custom_timestamp: pd.Timestamp = None) -> np.ndarray:
        """ Computes predicted potentials for all predictors for custom_timestamp (if provided) or current time. """
        potential_array = np.array([])
        for predictor in self.predictor_instances:
            try:
                predictor_input = self.get_predictor_input(predictor, custom_timestamp=custom_timestamp)
            except ValueError:
                continue  # happens if step is too low for rolling window
            prices, _ = predictor.predict(predictor_input)
            potential = (prices[-1] / predictor_input.iloc[
                -1] - 1).item()  # relative expected change at end of forecast horizon
            potential_array = np.append(potential_array, [potential])
        return potential_array

    @property
    def potential_estimates_dict(self) -> {str: np.ndarray}:
        """ Potential estimates array (one entry per predictor) with timestamp-keys (prediction times)."""
        if self._potential_estimates_dict is None:
            self._potential_estimates_dict = {}
            # precalculate potential estimates:
            print("Pre-calculating potential estimates for all timestamps and predictors...")
            for ind, timestamp in enumerate(tqdm(self.step_timestamp_list)):
                if timestamp is None: continue  # empty elements means no data for that date - skip
                # compute predicted potentials based on all predictors:
                self._potential_estimates_dict[timestamp] = self.compute_predicted_potentials(custom_timestamp=timestamp)

        return self._potential_estimates_dict

    @property
    def current_potential_estimates(self) -> np.ndarray:
        """ Returns potential estimates (relative expected return for next self.potential_horizon_days) per predictor. """
        if self.precalculate_predictor_observations:
            return self.potential_estimates_dict[self.current_step_timestamp]
        else:
            return self.compute_predicted_potentials()

    @property
    def current_observation(self):
        """ Construct and return current observable status. Includes predictors output, cash and holding. """
        observation = np.array([], dtype=np.int64)
        # add potential estimates:
        observation = np.append(observation, self.current_potential_estimates)
        # include cash and current holding:
        return np.append(observation, [self.cash, self.current_holding])

    @property
    def current_prices_per_product(self) -> pd.Series:
        """ Close prices of each product at the current step's time. """
        return self.product_set.price_frame.loc[self.current_step_timestamp, :]

    @property
    def current_balance(self) -> float:
        """ Current balance equals cash plus current holding. """
        return self.cash + self.current_holding

    @property
    def current_holding(self) -> float:
        """ Current holding equals current shares times current price. """
        return (self.shares_per_product * self.current_prices_per_product).sum().item()

    @property
    def open_positions(self) -> pd.DataFrame:
        """ Dataframe with all open positions' shares, leverages, prices and directions. """
        open_isins = list(self.shares_per_product[self.shares_per_product != 0].index)
        if len(open_isins) == 0:
            data_dict = {'Shares': None, 'Leverage': None, 'Price': None, 'Direction': None, '% Portfolio': None}
        else:
            data_dict = {'Shares': self.shares_per_product[open_isins],
                         'Leverage': self.product_set.leverage_frame.loc[self.current_step_timestamp, open_isins],
                         'Price': self.product_set.price_frame.loc[self.current_step_timestamp, open_isins],
                         'Direction': [self.product_set.by_isin[isin].direction for isin in open_isins],
                         '% Portfolio': self.shares_per_product[open_isins] * self.product_set.price_frame.loc[
                             self.current_step_timestamp, open_isins] / self.current_balance * 100,
                         }
        return pd.DataFrame(index=open_isins,
                            data=data_dict)

    def run_env_backtest(self,
                         agent: Union[MultiProductAgent, DQN],
                         mute_environment: bool = True,
                         print_statistics: bool = True,
                         reset_environment: bool = True,
                         track_portfolio_exposure_every: int = 15,
                         save_log_directory: str = None,
                         plot_results: bool = True,
                         performance_time_unit: Literal['p.a.', 'p.m.'] = 'p.m.',
                         sharpe_risk_free_rate_pa: float = .0278,
                         **plot_kwargs
                         ) -> (pd.Series, pd.Series, (float, float, float, float, float, float)):
        """
        Simulate agent behavior in environment.
        Returns tuple with
        1) policy_performance
        2) benchmark_performance as pd.Series normalised across performance_time_unit
        3) tuple with (mean, median, std.dev., min, max, SharpeRatio) alpha.
        """
        if reset_environment: obs, _ = self.reset()  # reset episode and fetch first observation
        else: obs = self.current_observation  # or fetch current observation

        if mute_environment: self.verbose = False

        ### iterate through all steps of environment:
        log = pd.DataFrame()  # initialise log frame
        for ind in tqdm(range(self.current_step, self.total_steps)):
            action, _ = agent.predict(obs)  # infer action
            if isinstance(action, np.ndarray): action = action.item()  # stable_baselines DQN returns np.array

            obs, _, done, truncated, info = self.step(action, track_portfolio_exposure=(
                        ind % track_portfolio_exposure_every == 0))  # retrieve new observation and info

            if done or truncated: break  # check whether episode is finished
            log = pd.concat([log, pd.DataFrame(info, index=[info['Step']])])  # log info

        # datetime-index for column concatenation and coherent structure:
        log['Time'] = pd.to_datetime(log['Time'])
        log.set_index('Time', inplace=True)
        log['Time'] = log.index  # include index also as column for e.g. groupby

        ### compute benchmark:
        benchmark = self.starting_cash / self.price_series.loc[log['Time'].iloc[0]] * self.price_series.loc[
            log['Time']]  # construct benchmark return if HODL
        log['Benchmark'] = benchmark

        ### annualized / monthly performance:
        # compute time-window normalized returns:
        if performance_time_unit != 'p.a.' and performance_time_unit != 'p.m.':
            raise ValueError("performance_time_unit must be 'p.a.' or 'p.m.'!")
        grouped_frame = log.loc[:, ['Total', 'Benchmark', 'Time']].groupby(
            log.index.year if performance_time_unit == 'p.a.' else [log.index.year, log.index.month]).agg(
            ['first', 'last'])
        normalised_benchmark_return_series = ((
                                                      (grouped_frame['Benchmark', 'last'] - grouped_frame[
                                                          'Benchmark', 'first'])  # increase
                                                      / grouped_frame['Benchmark', 'first']  # relative return
                                              ) * (
                                                      (grouped_frame['Time', 'last'] - grouped_frame[
                                                          'Time', 'first']).max()  # regular time duration
                                                      / (grouped_frame['Time', 'last'] - grouped_frame['Time', 'first'])
                                                  # actual time duration
                                              ))
        normalised_policy_return_series = ((
                                                   (grouped_frame['Total', 'last'] - grouped_frame[
                                                       'Total', 'first'])  # increase
                                                   / grouped_frame['Total', 'first']  # relative return
                                           ) * (
                                                   (grouped_frame['Time', 'last'] - grouped_frame[
                                                       'Time', 'first']).max()  # regular time duration
                                                   / (grouped_frame['Time', 'last'] - grouped_frame['Time', 'first'])
                                               # actual time duration
                                           ))
        # remove multi index and set time-index
        normalised_benchmark_return_series.index = grouped_frame['Time', 'first']
        normalised_policy_return_series.index = grouped_frame['Time', 'first']

        if save_log_directory:  # eventually save model
            log.to_csv(save_log_directory / filemgmt.file_title("Environment Backtest Log", dtype_suffix='.csv'))

        # compute portfolio statistics:
        mean_policy = normalised_policy_return_series.mean(); median_policy = normalised_policy_return_series.median()
        std_dev_policy = normalised_policy_return_series.std()
        min_policy = normalised_policy_return_series.min(); max_policy = normalised_policy_return_series.max()

        # compute alpha statistics:
        normalised_alpha_series = normalised_policy_return_series - normalised_benchmark_return_series
        mean_alpha = normalised_alpha_series.mean(); median_alpha = normalised_alpha_series.median()
        std_dev_alpha = normalised_alpha_series.std()
        min_alpha = normalised_alpha_series.min(); max_alpha = normalised_alpha_series.max()
        # normalise risk-free rate according to performance_time_unit
        risk_free_rate_normalized = sharpe_risk_free_rate_pa if performance_time_unit == 'p.a.' else (1 + sharpe_risk_free_rate_pa) ** (1/12) - 1
        sharpe_ratio = (mean_policy - risk_free_rate_normalized) / std_dev_policy

        if print_statistics:
            print(f"--------- Policy {performance_time_unit} Performance Statistics ---------")
            # policy statistics:
            print(f"Policy: \tMedian {round(median_policy*100, 3)}%\t\tMean {round(mean_policy*100, 3)}%\t\tStd.Dev. {round(std_dev_policy*100, 3)}%")
            print(f"\t\t\tMax. {round(max_policy*100, 3)}%\t\tMin. {round(min_policy*100, 3)}%")
            # alpha statistics:
            print(f"Alpha:\t\tMedian {round(median_alpha*100, 3)}%\t\tMean {round(mean_alpha*100, 3)}%\t\tStd.Dev. {round(std_dev_alpha*100, 3)}%")
            print(f"\t\t\tMax. {round(max_alpha*100, 3)}%\t\tMin. {round(min_alpha*100, 3)}%")
            print(f"Sharpe Ratio: {round(sharpe_ratio, 3)} - {f'Annual Sharpe Ratio: {round(sharpe_ratio * (12**(1/2)), 3)}' if performance_time_unit == 'p.m.' else ''}")
            overperform_mask = (normalised_alpha_series > 0)
            print(f"Over-performed {overperform_mask.value_counts()[True]} / {len(overperform_mask)} epochs")

        if plot_results:
            self.plot_backtest_results(log_df=log, policy_return_series=normalised_policy_return_series,
                                       benchmark_return_series=normalised_benchmark_return_series,
                                       performance_time_unit=performance_time_unit,
                                       agent=agent,
                                       **plot_kwargs)


        return normalised_policy_return_series, normalised_benchmark_return_series, (mean_alpha, median_alpha, std_dev_alpha, min_alpha, max_alpha, sharpe_ratio)

    def plot_backtest_results(self, log_df: pd.DataFrame,
                              policy_return_series: pd.Series = None,
                              benchmark_return_series: pd.Series = None,
                              performance_time_unit: Literal['p.a.', 'p.m.'] = 'p.a.',
                              agent: Union[MultiProductAgent, DQN] = None,
                              save_fig_directory: str = None):
        """
        Visualize backtest results.

        Include monthly or annual returns as policy_return_series to include time-normalized return plot.
        """
        # prepare axes:
        if policy_return_series is not None:
            fig, (return_ax, performance_ax, exposure_ax, cash_ax) = plt.subplots(4, 1, figsize=(16, 13))
        else:
            fig, (return_ax, exposure_ax, cash_ax) = plt.subplots(3, 1, figsize=(16, 13))
        potential_ax = exposure_ax.twinx()
        dates = pd.to_datetime(log_df.index)

        # portfolio performance:
        val_df = log_df.loc[:,
                 [f'Avg. Expected Potential / {self.potential_horizon_days}d', 'Total Exposure', 'Cash', 'Total']]
        normalized_policy_portfolio_value = log_df['Total'] / log_df['Benchmark'].iloc[0]
        normalized_benchmark_portfolio_value = log_df['Benchmark'] / log_df['Benchmark'].iloc[0]
        return_ax.plot(dates, normalized_policy_portfolio_value, color='orange', label='Policy')
        return_ax.plot(dates, normalized_benchmark_portfolio_value, color='blue', label='Benchmark')
        return_ax.set_title('Normalized Policy vs. Benchmark Portfolio Value')
        return_ax.set_xlabel('Date')
        return_ax.set_ylabel('Normalized Balance')
        return_ax.legend()
        return_ax.grid(True)

        # relative (time-normalised) performance plot:
        if policy_return_series is not None:
            performance_ax.plot(policy_return_series * 100, marker='o', color='orange', label='Policy')
            if benchmark_return_series is not None:
                performance_ax.plot(benchmark_return_series * 100, marker='o', color='blue', label='Benchmark')

                over_perform_mask = (policy_return_series > benchmark_return_series)
                plottable_over_perform_mask = over_perform_mask | np.roll(over_perform_mask, 1)  # extend each True to the next index
                # because each return value considers the return until the next value
                performance_ax.fill_between(policy_return_series.index, benchmark_return_series * 100,
                                            policy_return_series * 100,
                                            where=plottable_over_perform_mask,
                                            edgecolor='green', color='green', alpha=.5,
                                            label='Over-performance')

                under_perform_mask = (policy_return_series <= benchmark_return_series)
                plottable_under_perform_mask = under_perform_mask | np.roll(under_perform_mask, 1)  # extend each True to the next index
                performance_ax.fill_between(policy_return_series.index, benchmark_return_series * 100,
                                            policy_return_series * 100,
                                            where=plottable_under_perform_mask,
                                            edgecolor='red', color='red', alpha=.5,
                                            label='Under-performance')

                performance_ax.axhline(y=0, color='black')
            performance_ax.set_xlabel('Date')
            performance_ax.set_ylabel(f"Return {performance_time_unit} [%]")
            performance_ax.legend(loc='upper left')
            performance_ax.grid()

        # include potential thresholds:
        potentials = val_df[f'Avg. Expected Potential / {self.potential_horizon_days}d'] * 100
        if isinstance(agent, MultiProductAgent):
            # scale and derive thresholds:
            scaled_thresholds = [threshold * sign * 100 for threshold, sign in
                                 product(agent.abs_potential_threshold_steps, [-1, 1])]
            # plot threshold lines:
            for ind, scaled_threshold in enumerate(scaled_thresholds):
                potential_ax.axhline(y=scaled_threshold, color='purple',
                                     linestyle=':', alpha=.3,
                                     label='Action Thresholds' if ind == 0 else '_',
                                     # create legend entry only for first line
                                     )

            # avg potential:
            potentials = val_df[f'Avg. Expected Potential / {self.potential_horizon_days}d'] * 100
            buy_long_signals = np.where(potentials > scaled_thresholds[3], potentials, np.nan)
            potential_ax.plot(dates, buy_long_signals, color='green', label='Buy Long Signal')
            sell_short_signals = np.where((potentials > scaled_thresholds[1]) & (potentials < scaled_thresholds[3]),
                                          potentials, np.nan)
            potential_ax.plot(dates, sell_short_signals, color='lightgreen', label='Sell Short Signal')
            hold_signals = np.where((potentials > scaled_thresholds[0]) & (potentials < scaled_thresholds[1]),
                                    potentials, np.nan)
            potential_ax.plot(dates, hold_signals, color='grey', label='Hold Signal')
            sell_long_signals = np.where((potentials > scaled_thresholds[2]) & (potentials < scaled_thresholds[0]),
                                         potentials, np.nan)
            potential_ax.plot(dates, sell_long_signals, color='lightcoral', label='Sell Long Signal')
            buy_short_signals = np.where(potentials < scaled_thresholds[2], potentials, np.nan)
            potential_ax.plot(dates, buy_short_signals, color='red', label='Buy Short Signal')
        else:  # if agent doesn't follow a manual agenda:
            potential_ax.plot(dates, potentials, color='green', label='Expected Potential')

        # portfolio exposure:
        exposure_ax.plot(dates, val_df['Total Exposure'], color='black', label='Portfolio Exposure', marker='o',
                         markersize=5)

        # cash quote:
        cash_percent = (val_df['Cash'] / val_df['Total']) * 100
        cash_ax.plot(dates, cash_percent, color='black', label='Cash Quote [%]')
        cash_ax.fill_between(dates, 0, cash_percent, color='grey', alpha=.15, label='Cash')
        cash_ax.fill_between(dates, cash_percent, 100, color='purple', alpha=.15, label='Stock Holding')
        cash_ax.set_ylim([0, 100])
        cash_ax.set_xlabel('Date');
        cash_ax.set_ylabel('Share of Total Portfolio [%]')
        cash_ax.set_title('Portfolio Composition Cash vs. Holding')
        cash_ax.legend(loc='upper left');
        cash_ax.grid(True)

        # formatting:
        exposure_ax.set_xlabel('Date')
        exposure_ax.legend(loc='upper left');
        potential_ax.legend(loc='lower right')
        exposure_ax.set_ylabel('Total Exposure [x]');
        potential_ax.set_ylabel(f'Expected Forward Potential / {self.potential_horizon_days}d [%]')
        exposure_ax.set_ylim([-5, 5])
        max_potential = np.max(val_df[f'Avg. Expected Potential / {self.potential_horizon_days}d'] * 100);
        potential_ax.set_ylim([-100, 100])
        exposure_ax.set_title('Portfolio Exposure and Predicted Potential')
        exposure_ax.grid(True)

        fig.tight_layout()

        if save_fig_directory is not None:  # eventually save result plot:
            plt.savefig(save_fig_directory / filemgmt.file_title("Backtest Result Visualisation", ".png"))

        plt.show()