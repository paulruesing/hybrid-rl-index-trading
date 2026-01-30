from statistics import median

import src.pipeline.preprocessing as preprocessing
from src.pipeline.predictors import LSTMPredictor
from src.pipeline.financial_products import KOCertificate, KOCertificateSet
from src.utils import file_management as filemgmt
from src.pipeline.rl_agents import MultiProductAgent

from datetime import datetime
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
from typing import Literal
from datetime import datetime


# todo: implement action execution and set "completed" to True
class TradeImplementor:
    """
    TradeImplementor is a class designed to manage and log trade actions such as buying or selling assets.

    Methods
    -------
    __init__ :
        Initializes the TradeImplementor instance and sets up an empty action log.
    __call__(isin: str, amount: int, action: Literal['buy', 'sell']) :
        Captures and logs trade action details along with a timestamp.

    Parameters
    ----------
    __call__
        isin : str
            The International Securities Identification Number (ISIN) of the asset being traded.
        amount : int
            The amount of units being traded.
        action : Literal['buy', 'sell']
            Specifies the type of trade action. Must be either 'buy' or 'sell'.

    Attributes
    ----------
    action_log : dict
        A dictionary that stores logged trade actions. Each entry contains a timestamp as the key and a dictionary with trade details as the value.
    """
    def __init__(self):
        self.action_log = {}

    def __call__(self, isin: str, amount: int, action: Literal['buy', 'sell']):
        """
        Records an action (buy or sell) for a financial instrument with the given ISIN and amount into a log.

        Parameters
        ----------
        isin : str
            The International Securities Identification Number (ISIN) of the financial instrument being traded.
        amount : int
            The quantity of the financial instrument to be traded.
        action : Literal['buy', 'sell']
            The type of transaction to record, either 'buy' or 'sell'.
        """
        today = datetime.today().strftime('%Y-%m-%d %H_%M_%S')
        self.action_log[today] = {"isin": isin, "amount": amount, "action": action, "completed": False}


class RLTradingEnv(gym.Env):
    """
    A reinforcement learning environment for trading using structured price data and predictive signals.

    This environment wraps a trading simulation where an agent interacts with financial instruments
    (e.g., knockout certificates) and receives predictions from one or more LSTM-based predictors.
    Actions are taken based on leverage categories, and rewards are based on changes in portfolio value.

    The environment steps once per day for each daily prediction hour defined by the predictors.

    Attributes
    ----------
    metadata : dict
        Dictionary containing environment metadata, including render modes.
    price_sampling_rate_minutes : int
        Frequency of the price data in minutes.
    starting_cash : float
        Initial cash balance for the agent.
    commission_rate : float
        Effective commission rate or spread applied on each transaction.
    verbose : bool
        Whether to print detailed step and trade information.
    potential_horizon_days : int
        Forecast days to scale expected return potential to. Only relevant for info statements.
    current_episode : int
        Current episode number during the simulation.
    predictor_instances : list of LSTMPredictor
        Predictors used to generate potential signals.
    _daily_prediction_hours : object
        Internal predictor setting for daily prediction sampling.
    _product_set : KOCertificateSet
        Set of financial products (e.g., long/short leverage instruments).
    _leverage_categories : list of float
        Leverage brackets used to define the action space.
    _include_open_leverage_category : bool
        Whether an additional category is included for leverage greater than the last entry.
    trading_quantity_per_leverage_factor : float
        Determines trade size as a fraction of balance per leverage category.
    sell_opposite_direction_if_no_cash : bool
        If insufficient cash for buying operation, specifies whether to sell opposite certificates.
    sell_all_opposite_products_if_no_cash : bool
        If sell_opposite_products_if_no_cash is True and this is True, sells all opposite products;
        otherwise, sells only opposite leverage of products.
    _action_enum : object
        Internal enumeration representation for actions.
    action_space : Discrete
        Action space for the environment, determined by leverage categories and product types.
    _step_dates_list : list
        List of step dates used during the simulation.
    _step_timestamp_list : list
        Internal list tracking timestamps of each step.
    current_step : int
        Current step number during the episode.
    observation_space : Box
        Observation space for the environment, containing predictor outputs and portfolio state.
    _potential_estimates_dict : dict
        Internal cached dictionary of potential observations from predictors.
    precalculate_predictor_observations : bool
        If true, calculates all potential estimates upon accessing the first observation for runtime efficiency.
    reward_range : tuple of float
        Possible range of rewards for actions, currently set to (-infinity, infinity).
    cash : float
        Current cash balance available to the agent.
    shares_per_product : Series
        Pandas Series representing the number of shares owned per product by ISIN.
    shares : float
        Redundant attribute for tracking shares (to be removed in the future).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 price_csv_path: Union[str, Path] = None,
                 price_series: pd.Series = None,
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
                 sell_opposite_direction_if_no_cash: bool = True,
                 sell_all_opposite_products_if_no_cash: bool = True,
                 starting_cash=1000000,
                 commission_rate=0.001,  # reflects a typical spread, because we can trade commission-free through wikifolio
                 verbose=True,
                 potential_horizon_days: int = 90,
                 ):
        """
        Initializes an environment for reinforcement learning-based trading. This class is designed to work with a
        price time series or price CSV file, multiple predictors, and a set of financial products to create an
        actionable trading environment. It allows the customization of trading parameters such as commission rates,
        leverage categories, and cash management rules. The environment provides an observation space based on
        predictor outputs and financial metrics, and an action space derived from financial product configurations.

        Parameters
        ----------
        price_csv_path : Union[str, Path], optional
            Path to a CSV file containing price data. Either `price_csv_path` or `price_series` must be provided.
        price_series : pd.Series, optional
            Time series object containing price data. Either `price_series` or `price_csv_path` must be provided.
        price_sampling_rate_minutes : int, optional
            Frequency, in minutes, at which prices are sampled for the environment. Default is 1 minute.
        price_column : str, optional
            Column name in the price dataset to be used as the price. Default is 'close'.
        date_column : str, optional
            Column name in the price dataset used for dates. Default is 'date'.
        predictor_instances : list of LSTMPredictor, optional
            List of predictor instances that provide features for the environment.
        precalculate_predictor_observations : bool, optional
            Whether to precalculate predictor observations for optimization. Default is True.
        product_set : KOCertificateSet, optional
            Set of financial products (e.g., knockout certificates) used to create the action space.
        leverage_categories : list of float, optional
            List of leverage factors used to define the action space. Default is [1.0, 2.0, 3.0, 4.0, 5.0].
        include_open_leverage_category : bool, optional
            Whether to include an open leverage category where larger values than the last entry represent the
            highest category. Default is False.
        trading_quantity_per_leverage_factor : float, optional
            Multiplier to adjust trading size according to the leverage factor. Default is 1.0.
        sell_opposite_direction_if_no_cash : bool, optional
            Whether to liquidate positions in the opposite direction if there is no cash available for trading.
            Default is True.
        sell_all_opposite_products_if_no_cash : bool, optional
            Whether to sell all products in the opposite direction if there is no cash available for trading.
            Default is True.
        starting_cash : float, optional
            Initial amount of cash in the environment. Default is 1,000,000.
        commission_rate : float, optional
            Rate of commission or transaction fee, typically reflecting the spread. Default is 0.001.
        verbose : bool, optional
            Enables detailed logs and debug messages during environment interaction. Default is True.
        potential_horizon_days : int, optional
            The maximum potential horizon in days for the trading environment. Default is 90.

        Attributes
        ----------
        _price_series : pd.Series or None
            Stores the provided price series data.
        _price_csv_path : Union[str, Path] or None
            Path to the provided price CSV file.
        _price_column : str
            The name of the column used for price data.
        _date_column : str
            The name of the column used for date data.
        price_sampling_rate_minutes : int
            Frequency, in minutes, at which prices are sampled.
        starting_cash : float
            Initial cash amount in the environment.
        commission_rate : float
            Transaction fee or spread rate for trading.
        verbose : bool
            Flag to toggle detailed logging.
        potential_horizon_days : int
            Maximum number of allowable trading days.
        current_episode : int
            Counter for the current episode in the environment.
        predictor_instances : list of LSTMPredictor or None
            List of predictors used in the environment.
        _daily_prediction_hours : None or any
            Reserved for configurations related to prediction sampling.
        _product_set : KOCertificateSet or None
            Set of financial products defining the action space.
        _leverage_categories : list of float
            Leverage values used in the action space.
        _include_open_leverage_category : bool
            Indicates if the highest leverage category is open.
        trading_quantity_per_leverage_factor : float
            Determines trade size based on leverage.
        sell_opposite_direction_if_no_cash : bool
            Indicates if opposite positions will be liquidated when cash is insufficient.
        sell_all_opposite_products_if_no_cash : bool
            Indicates if all opposite positions will be liquidated when cash is insufficient.
        _action_enum : None or any
            Represents the action space structure.
        action_space : gym.spaces.Discrete
            Discrete action space for the trading environment.
        _step_dates_list : None or any
            Contains the list of trading step dates (to be calculated dynamically).
        _step_timestamp_list : None or any
            Contains the list of trading step timestamps (to be calculated dynamically).
        current_step : int
            The current step within a trading episode.
        observation_space : gym.spaces.Box
            Continuous observation space defined by predictors and financial metrics.
        _potential_estimates_dict : None or any
            Holds cached potential estimates for predictors.
        precalculate_predictor_observations : bool
            Flag to control predictor observation precomputation.
        reward_range : tuple of float
            Specifies the range of possible reward values.
        cash : float
            Current cash amount in the environment.
        shares_per_product : pd.Series
            Tracks the number of shares held for each product.
        shares : float
            Tracks the total number of all shares collectively held.
        """
        super(RLTradingEnv, self).__init__()  # initialise base class

        if price_series is None and price_csv_path is None:
            raise ValueError("Either price_series or price_csv_path must be provided.")
        self._price_series = price_series
        self._price_csv_path = price_csv_path
        self._price_column = price_column
        self._date_column = date_column

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

    def step(self, action: int, track_portfolio_exposure: bool = True, start_new_episode_if_finished: bool = True,
             trade_implementation_callback: callable = None):
        """
        Performs one step in the environment simulation, taking the specified action and advancing
        the simulation. Updates the internal state, computes the reward, and provides relevant
        information about the environment's current state.

        Parameters
        ----------
        action : int
            The action to be executed, represented as an integer corresponding to an internal
            action enumeration dictionary.
        track_portfolio_exposure : bool, optional
            Flag indicating whether to calculate and track the portfolio's current exposure
            during this step (default is True).
        start_new_episode_if_finished : bool, optional
            Flag specifying whether to automatically start a new episode if the current
            episode has finished. If set to False and the episode is finished, a ValueError
            will be raised (default is True).

        Returns
        -------
        obs : object
            The current observation of the environment's state after taking the specified
            action.
        reward : float
            The computed reward for the step, typically representing the change in the
            portfolio balance resulting from the action taken.
        done : bool
            A flag indicating whether the current episode has finished.
        truncated : bool
            Gymnasium-specific flag, currently always set to False for this implementation.
        info : dict
            Additional information about the environment's current state after the step. The
            dictionary may include keys such as:
                - 'Step': Current step number.
                - 'Time': Timestamp of the current step in '%Y-%m-%d %H:%M:%S' format.
                - 'Reward': The rounded reward for the step.
                - 'Action': Action performed, as defined in the action enumeration dictionary.
                - 'Avg. Expected Potential / {horizon}d': Average scaled predicted potential
                  over the defined horizon.
                - 'Cash': The current cash balance available in the portfolio.
                - 'Total': The total balance of the portfolio, including cash and investments.
                - 'Total Exposure' (if tracked): Weighted average exposure of the portfolio
                  based on leverage, direction, and portfolio share.

        Raises
        ------
        ValueError
            Raised when `start_new_episode_if_finished` is False, but the current episode
            has ended, and a new episode cannot be started.
        """
        # infer current balance:
        balance = self.current_balance

        # increase step
        self.current_step += 1
        if self.current_step == self.total_steps:  # if episode is finished
            done = True
            if start_new_episode_if_finished:
                self.current_episode = self.next_episode()
                self.init_start_step()
            else:
                raise ValueError("Episode is finished, but start_new_episode_if_finished is False. Aborting...")
        else:
            done = False

        # take action:
        self.take_action(action, trade_implementation_callback=trade_implementation_callback)

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

        # format reward:
        if isinstance(reward, (np.float32, np.float64, np.ndarray)):
            formatted_reward = reward.item()  # safely get a Python float
        else: formatted_reward = reward
        formatted_reward = round(formatted_reward, 2) if (formatted_reward is not np.nan) and (formatted_reward != 0) else 0

        # todo: if tendencies included in observations, include such here
        # construct info dictionary:
        info = {'Step': self.current_step,
                'Time': self.current_step_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Reward': formatted_reward,
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

    def take_action(self, action: int, trade_implementation_callback: callable = None):
        """
        Executes trading actions such as buying or selling products based on specified leverage,
        direction, and other rules. Determines the appropriate products and quantities for the action
        and updates the portfolio accordingly.

        Parameters
        ----------
        action : int
            The action identifier used to determine the type of operation (e.g., Buy or Sell),
            the direction (e.g., long or short), and the leverage span.

        Notes
        -----
        - The method determines which product to trade based on its leverage, direction, and a predefined
          mapping (`self.action_enum_dict`). If the desired leverage is unavailable, it attempts to find
          the next suitable leverage by modifying the search criteria.
        - The operation (Buy or Sell) takes into account the available cash, portfolio balance, and trading
          limits. For buying actions, it calculates the maximum number of shares that can be purchased
          within the provided constraints. For selling actions, it identifies the positions to close
          using the specified leverage and direction.
        - For `Buy` actions, if insufficient cash is available, there are options (controlled by flags)
          to sell opposite-direction products to free up cash for buying.
        - Verbose logging is provided if the `self.verbose` flag is enabled, offering insight into
          each operation's details and actions taken.
        - Commission rates are factored into both buying and selling price calculations.
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

            # if purchase not possible:
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

            else:  # if purchase possible
                if self.verbose:
                    print(
                        f"[STEP {self.current_step}] Bought {shares_to_buy} shares of {isin} ({direction} with leverage in {leverage_span}) at {price}.")
                    print(f"    Cash: {self.cash}, Holding: {self.current_holding}")

                # action callback:
                if trade_implementation_callback is not None:
                    trade_implementation_callback(isin = isin, amount = shares_to_buy, action = type.lower())

        if type == 'Sell':
            # select all products with leverages inside span and higher:
            # todo: reflect whether all products with higher leverages should remain, or whether only inside span should be sold
            try:
                open_candidates = self.open_positions.loc[self.open_positions.Direction == direction]
            except KeyError:  # some data couldn't be fetched
                print(f"Problem while calculating open positions for {direction} at {self.current_step_timestamp}. Continuing.")
                return   # aka. no product found

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

            # action callback:
            if trade_implementation_callback is not None:
                if isinstance(shares_to_sell, pd.Series):  # if multiple shares to sell (can happen upon selling)
                    for isin, shares in zip(isins, shares_to_sell):
                        trade_implementation_callback(isin=isin, amount=shares, action=type.lower())

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
        Resets the environment for a new episode.

        This method resets the internal state of the environment, including cash, shares held,
        and observation, to prepare it for a new trading episode. It also increments the
        internal episode counter and provides an option for reproducibility by setting a
        random seed. An optional status message is returned for informational purposes.

        Parameters
        ----------
        seed : int, optional
            A seed value to initialize the random number generator. Default is None.

        Returns
        -------
        tuple of (numpy.ndarray, dict)
            The updated observation array representing the initial state of the environment
            after reset, and a dictionary containing a status message.

        Notes
        -----
        If `verbose` is enabled, the method prints a message stating the start of the new
        trading episode. This can be useful for monitoring progress during simulations.
        """
        self.init_start_step()
        if self.verbose:  # info statement
            print(f"Starting episode {self.current_episode + 1}", (f"/ {self.total_episodes}"))
        self.cash = self.starting_cash
        self.shares_per_product.iloc[:] = 0
        return self.current_observation, {'Status': 'Starting new episode'}  # info dict

    def get_predictor_input(self, predictor: LSTMPredictor, custom_timestamp: pd.Timestamp = None) -> pd.Series:
        """
        Calculates a series of sampled prices from the price series based on the rolling window size, sampling rate,
        and an optional custom timestamp. This data preparation is used for prediction purposes.

        Parameters
        ----------
        predictor : LSTMPredictor
            The predictor instance that provides configuration values like rolling window size, sampling rate,
            and predict-before-hour settings.
        custom_timestamp : pd.Timestamp, optional
            A custom timestamp to use for slicing the series instead of the current step's timestamp.

        Returns
        -------
        pd.Series
            A series of sampled prices representing the historical price data over a sliding window,
            adjusted by the sampling rate and prediction hour configuration.
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

        # assertion to check sampling rate and length of sampled_prices matches the predictor's specs
        assert len(sampled_prices) == rolling_window_size, (
            f"Expected {rolling_window_size} samples, but got {len(sampled_prices)}"
        )
        """assert (sampled_prices.index[1] - sampled_prices.index[0]).total_seconds() / 60 == sampling_rate_minutes, (
            f"Expected sampling rate of {sampling_rate_minutes} minutes, "
            f"but got {(sampled_prices.index[1] - sampled_prices.index[0]).total_seconds() / 60} minutes"
        )"""  # todo: ponder better idea, this doesn't work if e.g. weekends are included

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
        """ product_set setter. """
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
        for horizon_minutes, potential in zip(self.observation_horizons_minutes, self.current_potential_estimates):
            # todo: think how to change this 24 to 14 (business day duration) because observation_horizons_minutes uses such indirectly
            potential = potential / horizon_minutes * self.potential_horizon_days * 24 * 60  # scale per minute and then to per self.potential_horizon_days days

            potential_list = np.append(potential_list, [potential])
        return np.nanmean(potential_list).item()

        potential_list = np.array([])
        for type, horizon_minutes, potential in zip(self.observation_types, self.observation_horizons_minutes, self.current_observation):
            if type != 'potential': continue
            # todo: think whether this 24 needs to 14 (business day duration) because observation_horizons_minutes uses such indirectly
            # likely should be, but than the manual agent needs to be reworked!!! (by a calculatable factor)
            potential = potential / horizon_minutes * self.potential_horizon_days * 24 * 60  # scale per minute and then to per self.potential_horizon_days days
            potential_list = np.append(potential_list, [potential])
        return np.nanmean(potential_list).item()

    @property
    def price_series(self):
        """ Underlying price series for observations and reward calculations. """
        if self._price_series is None:
            self._price_series = preprocessing.read_price_csv(csv_path=self.price_csv_path, date_column=self.date_column,
                                                                price_column=self.price_column)
        return self._price_series
    @price_series.setter
    def price_series(self, value: pd.Series):
        """ Underlying price series for observations and reward calculations. """
        try:  # check if all previous datapoints are equal
            pure_extension_bool = (self._price_series == value.iloc[:len(self._price_series)]).all()
        except ValueError:
            pure_extension_bool = False

        # if not sole extension of price series but value change -> reset potential estimates
        if not pure_extension_bool: self._potential_estimates_dict = None
        else:
            if self.verbose: print("Price series has been extended. Previous potential estimates remain unchanged.")

        self._price_series = value
        self._step_dates_list = self._step_timestamp_list = None

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
        """
        Computes predicted relative potentials for each predictor. Leverages self.compute_predicted_prices().

        Parameters
        ----------
        custom_timestamp : pd.Timestamp, optional
            A specific timestamp to compute predicted potentials for. If not provided, default behavior will be used.

        Returns
        -------
        np.ndarray
            An array of predicted potential values, representing the relative expected change at the end of the forecast horizon.
        """
        potential_array = np.array([])
        for _, (pred_input, prices) in self.compute_predicted_prices(custom_timestamp=custom_timestamp).items():
            potential = (prices.iloc[-1] / pred_input.iloc[
                -1] - 1).item()  # relative expected change at end of forecast horizon
            potential_array = np.append(potential_array, [potential])
        return potential_array

    def compute_predicted_prices(self, custom_timestamp: pd.Timestamp = None) -> dict:
        """
        Compute predicted prices for each predictor instance.

        This method iterates over all predictor instances associated with the object
        and computes predicted prices based on their specific prediction logic. The
        predicted prices are returned as a dictionary where the keys are the names
        of the predictors and the values are tuples containing the predictor inputs,
        a pandas Series of predicted prices, and corresponding timestamp indices.

        In case certain conditions, such as a too-low step for rolling window calculations,
        are not met, the corresponding predictors are skipped.

        Parameters
        ----------
        custom_timestamp : pd.Timestamp, optional
            A custom timestamp to override the default timestamp used for computing
            predictor inputs. If not provided, defaults to None.

        Returns
        -------
        dict
            A dictionary mapping predictor names to tuples containing:
            1. Predictor input.
            2. A pandas Series with the predicted prices as data and respective dates as the index.
        """
        pred_feature_target_dict = {}
        for predictor in self.predictor_instances:
            try:
                predictor_input = self.get_predictor_input(predictor, custom_timestamp=custom_timestamp)
            except ValueError:
                continue  # happens if step is too low for rolling window

            prices, dates = predictor.predict(predictor_input, dtype='numpy')
            pred_feature_target_dict[predictor.name] = (predictor_input,
                                                        pd.Series(data=prices, index=dates))
        return pred_feature_target_dict

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
            try:  # try accessing precalculated estimates
                return self.potential_estimates_dict[self.current_step_timestamp]
            except KeyError:  # if dict doesn't include Key, manually calculate potential
                potential = self.compute_predicted_potentials()
                self._potential_estimates_dict[self.current_step_timestamp] = potential  # extend dict
                return potential  # and return manually calculated potential
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
        if hasattr(self, '_cached_open_positions'):  # use cached open positions
            if self._cached_open_positions_ts == self.current_step_timestamp:
                return self._cached_open_positions

        open_isins = list(self.shares_per_product[self.shares_per_product != 0].index)
        if len(open_isins) == 0:
            data_dict = {'Shares': None, 'Leverage': None, 'Price': None, 'Direction': None, '% Portfolio': None}
        else:
            data_dict = {}  # initialise data dict (this happens separately to time the value composition below)
            data_dict['Shares'] = self.shares_per_product[open_isins]

            # previous (VERY EXPENSIVE, ~100ms): data_dict['Leverage'] = self.product_set.leverage_frame.loc[self.current_step_timestamp, open_isins]
            # now (QUITE FAST, ~7ms):
            leverage_list = []  # initialise separately, to catch errors within for loop
            for isin in open_isins:
                try:
                    leverage_list.append(self.product_set.by_isin[isin].leverage_series.loc[self.current_step_timestamp])
                except KeyError:  # time point not found
                    print(f"Error with leverage calculation for {isin} at {self.current_step_timestamp}!")
                    leverage_list.append(0.0)
            data_dict['Leverage'] = leverage_list

            data_dict['Price'] = self.product_set.price_frame.loc[self.current_step_timestamp, open_isins]
            data_dict['Direction'] = [self.product_set.by_isin[isin].direction for isin in open_isins]
            data_dict['% Portfolio'] = self.shares_per_product[open_isins] * self.product_set.price_frame.loc[
                             self.current_step_timestamp, open_isins] / self.current_balance * 100

        # cache result:
        self._cached_open_positions = pd.DataFrame(data_dict, index=open_isins)
        self._cached_open_positions_ts = self.current_step_timestamp

        return pd.DataFrame(index=open_isins,
                            data=data_dict)

    def run_env_backtest(self,
                         agent: Union[MultiProductAgent, DQN],
                         mute_environment: bool = True,
                         print_statistics: bool = True,
                         reset_environment: bool = True,
                         track_portfolio_exposure_every: int = 45,
                         save_log_directory: str = None,
                         plot_results: bool = True,
                         performance_time_unit: Literal['p.a.', 'p.m.'] = 'p.m.',
                         sharpe_risk_free_rate_pa: float = .0278,
                         **plot_kwargs
                         ) -> (pd.Series, pd.Series, (float, float, float, float, float, float)):
        """
        Executes a backtesting procedure for a trading environment where an agent interacts with a market simulator
        to evaluate its performance. Tracks portfolio exposure, computes normalized returns, and benchmarks against
        a hold strategy (HODL). Computes performance statistics such as mean, median, standard deviation, Sharpe ratio,
        and alpha. Optionally logs results, plots performance metrics, and prints detailed statistics.

        Parameters
        ----------
        agent : Union[MultiProductAgent, DQN]
            The agent responsible for predicting actions during the backtesting process.
        mute_environment : bool, optional
            If True, suppresses verbose logging from the environment during backtesting.
        print_statistics : bool, optional
            If True, prints the performance statistics for the trading strategy over the backtesting period.
        reset_environment : bool, optional
            If True, resets the environment to the initial state before starting backtest iterations.
        track_portfolio_exposure_every : int, optional
            The number of steps after which the portfolio exposure should be tracked and updated.
        save_log_directory : str, optional
            Directory to save the backtesting log as a CSV file. If None, logging results are not saved.
        plot_results : bool, optional
            If True, visualizes the backtesting results using predefined plotting methods.
        performance_time_unit : Literal['p.a.', 'p.m.'], optional
            Specifies the performance evaluation time unit. 'p.a.' for annualized metrics and 'p.m.' for monthly metrics.
        sharpe_risk_free_rate_pa : float, optional
            Risk-free rate used for Sharpe ratio calculation, expressed as an annualized rate.
        plot_kwargs : dict, optional
            Additional plotting parameters passed as keyword arguments to the plot function.

        Returns
        -------
        pd.Series
            A pandas Series containing the normalized policy return series over the backtesting period.
        pd.Series
            A pandas Series containing the normalized benchmark return series (HODL) over the backtesting period.
        tuple of float
            A tuple containing alpha performance statistics: mean alpha, median alpha, standard deviation of alpha,
            minimum alpha, maximum alpha, and Sharpe ratio.

        """
        if reset_environment: obs, _ = self.reset()  # reset episode and fetch first observation
        else: obs = self.current_observation  # or fetch current observation

        if mute_environment: self.verbose = False

        ### iterate through all steps of environment:
        log_list = []  # initialise log list
        print('Stepping environment along complete provided price data...')
        for ind in tqdm(range(self.current_step, self.total_steps)):
            action, _ = agent.predict(obs)  # infer action

            if isinstance(action, np.ndarray): action = action.item()  # stable_baselines DQN returns np.array

            obs, _, done, truncated, info = self.step(action, track_portfolio_exposure=(
                        ind % track_portfolio_exposure_every == 0))  # retrieve new observation and info

            if done or truncated: break  # check whether episode is finished
            log_list.append(info)

        # convert log_list to dataframe:
        log = pd.DataFrame(log_list)
        log.set_index('Step', inplace=True)

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
        Plots the backtest results, including normalized portfolio values, performance metrics, portfolio
        exposure, cash distributions, and potential signals. This visualization provides detailed insights
        into the performance of a policy against a benchmark, highlighting areas of overperformance or
        underperformance and their respective signals.

        Parameters
        ----------
        log_df : pd.DataFrame
            DataFrame containing detailed portfolio logs, including potential values, total exposure,
            cash values, and portfolio totals such as "Total" and "Benchmark".

        policy_return_series : pd.Series, optional
            Time series representing portfolio returns calculated from the agent's policy results.
            Default is None.

        benchmark_return_series : pd.Series, optional
            Time series representing benchmark portfolio returns for comparison against the policy.
            Default is None.

        performance_time_unit : {'p.a.', 'p.m.'}, optional
            String indicating the time normalization for return performance, either per annum ('p.a.') or
            per month ('p.m.'). Default is 'p.a.'.

        agent : MultiProductAgent or DQN, optional
            The agent whose performance is being analyzed. If `MultiProductAgent`, action thresholds
            and signals will also be plotted. Default is None.

        save_fig_directory : str, optional
            File directory path to save the resulting visualized backtest plot as a PNG file. If not
            provided, the resulting visualization will not be saved. Default is None.
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

    def plot_current_predictions(self, mpl_palette: str = 'Set1', zoom_on_last_n_days: int = 60,
                                 displayed_potential_tolerance: float = 0.5,
                                 save_fig_directory: Union[str, Path] = None,
                                 hidden: bool = False):
        print('Creating the current prediction plot...')
        price_dict = self.compute_predicted_prices()
        ##### prepare plots:
        fig, axs = plt.subplots(
            len(price_dict) + 1,  # an extra row for each predictor
            1, figsize=(10, 15),
            gridspec_kw={'height_ratios': [3] + [1] * len(self.predictor_instances)}  # compound upper plot 3x height
        )
        compound_ax = axs[0];
        compound_ax.set_title('All Predictors')
        cmap = plt.colormaps[mpl_palette]
        color_list = [cmap(i) for i in range(len(price_dict))]

        # predictor with smallest timescale:
        finest_predictor_ind = np.argmin([pred.sampling_rate_minutes for pred in self.predictor_instances])
        displayed_features = None  # will be efficiently inferred in for loop

        ##### single prediction plotting:
        for pred_ind, (pred_name, (features, prediction)) in enumerate(price_dict.items()):
            # compound ax plot:
            if pred_ind == finest_predictor_ind:  # features only for predictor with finest time-resolution
                compound_ax.plot(features.index, features, marker='o', markevery=[-1], color='darkblue')
                displayed_features = features.copy()
            compound_ax.plot(prediction.index, prediction, marker='o', markevery=[-1], color=color_list[pred_ind],
                             label=pred_name, linestyle='dashed')

            # single ax plot:
            pred_ax = axs[pred_ind + 1]
            pred_ax.plot(features.index, features, marker='o', markevery=[-1], color='darkblue', label='Features')
            pred_ax.plot(prediction.index, prediction, marker='o', markevery=[-1], color=color_list[pred_ind],
                         label='Predictions', linestyle='dashed')
            pred_ax.set_title(pred_name)

        ##### include average potential in compound plot:
        if displayed_features is None:
            displayed_features = features.copy()
        avg_predicted_price = displayed_features.iloc[-1] * (1 + self.current_avg_scaled_predicted_potential)
        # if less or more of the potential is realised:
        prediction_cone = (displayed_features.iloc[-1] * (
                    1 + self.current_avg_scaled_predicted_potential * (1 - displayed_potential_tolerance)),
                           displayed_features.iloc[-1] * (1 + self.current_avg_scaled_predicted_potential * (
                                       1 + displayed_potential_tolerance)))
        # assert that first element is the smaller one:
        prediction_cone = (min(prediction_cone), max(prediction_cone))

        avg_predicted_price_date = displayed_features.index.max() + pd.Timedelta(days=self.potential_horizon_days)
        plot_dates = [displayed_features.index.max(), avg_predicted_price_date]
        last_feature_value = displayed_features.iloc[-1]
        compound_ax.plot(plot_dates,
                         [last_feature_value, avg_predicted_price], marker='X', color='Blue',
                         label='Average Predicted Potential')
        compound_ax.fill_between(
            x=plot_dates,
            y1=[last_feature_value, prediction_cone[0]],
            y2=[last_feature_value, prediction_cone[1]],
            color='blue',
            alpha=0.2,
            label='Prediction Cone'
        )

        ##### eventual zoom-in:
        if zoom_on_last_n_days is not None:
            # adjust x-axis limits:
            start_date = datetime.now() - pd.Timedelta(days=zoom_on_last_n_days)
            compound_ax.set_xlim(start_date, None)

            # filter the data within the new x_lim
            all_values = pd.concat(
                [tuple[0] for tuple in price_dict.values()] + [tuple[1] for tuple in price_dict.values()])
            visible_data = all_values[(all_values.index >= start_date)]

            # compute new y_lim based on the visible data and prediction cone:
            try:
                new_y_min = min(visible_data.min().min(), prediction_cone[0])
                new_y_max = max(visible_data.max().max(), prediction_cone[1])
            except AttributeError:
                new_y_min = min(visible_data.min(), prediction_cone[0])
                new_y_max = max(visible_data.max(), prediction_cone[1])
            compound_ax.set_ylim(new_y_min - (new_y_max - new_y_min) * .05,
                                 new_y_max + (new_y_max - new_y_min) * .05)

        ##### formatting:
        for ax in axs:
            ax.set_ylabel('Prices [USD]')
            if ax == axs[-1]: ax.set_xlabel('Date')
            ax.legend()
            ax.grid()
        fig.tight_layout()

        ##### display and eventual saving:
        if save_fig_directory is not None:  # eventually save result plot:
            plt.savefig(save_fig_directory / filemgmt.file_title("Prediction Visualisation", ".png"))
        plt.show()

####### Auxiliary Backtesting Functions #######
def env_parametrisation_loop(env_price_sampling_rate_minutes: int,
                             env_product_set: KOCertificateSet,
                             env_price_file_path: Path = None,
                             env_price_series: pd.Series = None,
                             sort_metric: Literal['Mean', 'Median', 'StdDev', 'Min', 'Max', 'SharpeRatio'] = 'Mean',
                             print_progress: bool = True,
                             backtest_database_dir: Union[str, Path] = None,
                             include_constant_params_in_output: bool = True,
                             track_portfolio_exposure_every: int = 100,
                             **param_grid_and_constants):
    """
    Executes a parameterization loop over varying and constant parameters for environment testing,
    evaluating backtesting performance metrics for different parameter configurations.

    This function enables systematic experimentation across different combinations of parameters.
    It validates the input data sources, iterates through the Cartesian product of parameter grids,
    executes backtests using the environment and agent, and compiles the resulting metrics into
    a structured DataFrame. If a directory for results is provided, the results can optionally
    be combined with existing results in the database.

    Parameters
    ----------
    env_price_sampling_rate_minutes : int
        The sampling rate (in minutes) for the price data in the environment.

    env_product_set : KOCertificateSet
        The product set object representing the financial instruments used in the backtest.

    env_price_file_path : Path, optional
        The file path to a CSV containing price data for environment backtests. Either this
        or `env_price_series` must be provided. Default is None.

    env_price_series : pd.Series, optional
        A pandas Series object containing price data for backtests. Either this or
        `env_price_file_path` must be provided. Default is None.

    sort_metric : Literal['Mean', 'Median', 'StdDev', 'Min', 'Max', 'SharpeRatio'], optional
        A string specifying the backtest metric by which to sort the final DataFrame. The default
        is 'Mean'.

    print_progress : bool, optional
        Whether to print progress information during the parameterization loop. Default is True.

    backtest_database_dir : Union[str, Path], optional
        Directory path for storing cumulative backtest results. If specified, results are appended
        to this database. Default is None.

    include_constant_params_in_output : bool, optional
        Whether to include constant parameter values in the result DataFrame. Default is True.

    **param_grid_and_constants : dict
        Keyword arguments representing parameters for testing. Iterable values are considered grid
        parameters (varying between backtests), while other values are treated as constants.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing backtest results. Each row represents a parameter configuration
        and its corresponding metrics. The DataFrame includes columns for grid parameters, metrics,
        and optionally constant parameters.

    """
    # Validate that at least one price source is provided
    if env_price_file_path is None and env_price_series is None:
        raise ValueError("Either 'env_price_file_path' or 'env_price_series' must be provided.")

    # Identify grid parameters (iterables) and constant parameters
    grid_params = {k: v for k, v in param_grid_and_constants.items() if isinstance(v, list)}
    constant_params = {k: v for k, v in param_grid_and_constants.items() if k not in grid_params}

    # Result frame with columns for all varying parameters and resulting metrics:
    columns = list(grid_params.keys()) + ['Mean', 'Median', 'StdDev', 'Min', 'Max', 'SharpeRatio']
    if include_constant_params_in_output: columns += list(constant_params.keys())
    result_array = []  # Initialize result array

    # All possible ordered pairs of grid parameters:
    n_configs = len(list(product(*grid_params.values())))
    for config_ind, config in enumerate(product(*grid_params.values())):
        if print_progress: print(f"\n--------- Backtest Config {config_ind + 1} / {n_configs} ---------")

        params = dict(zip(grid_params.keys(), config))
        # Add relevant parameters to env_kwargs
        env_kwargs = {
            **params,
            **constant_params,
            'verbose': False,
            'price_sampling_rate_minutes': env_price_sampling_rate_minutes,
            'product_set': env_product_set,
        }

        # Include either `env_price_file_path` or `env_price_series` in `env_kwargs`
        if env_price_series is not None:
            env_kwargs['price_series'] = env_price_series
        else:
            env_kwargs['price_csv_path'] = env_price_file_path

        agent_kwargs = {}
        try:  # Check for agent-specific parameters in env_kwargs
            agent_kwargs['abs_potential_threshold_steps'] = env_kwargs['abs_potential_threshold_steps']
            del env_kwargs['abs_potential_threshold_steps']
        except KeyError:
            pass

        try:
            temp_env = RLTradingEnv(**env_kwargs)

            temp_agent = MultiProductAgent(**agent_kwargs,
                                           # Infer other parameters from environment
                                           observation_types=temp_env.observation_types,
                                           observation_horizons_minute=temp_env.observation_horizons_minutes,
                                           n_leverage_categories=len(temp_env.leverage_categories),
                                           include_open_leverage_category=temp_env.include_open_leverage_category,
                                           potential_treshold_horizon_days=temp_env.potential_horizon_days)

            # Settings and losses
            _, _, metric_tuple = temp_env.run_env_backtest(temp_agent, mute_environment=True, reset_environment=True,
                                                           track_portfolio_exposure_every=track_portfolio_exposure_every,
                                                           plot_results=False)
        except (ValueError, AttributeError) as e:
            print("Caught an exception:", e)
            metric_tuple = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        # Convert np values to regular values and concat to result row:
        metric_list = [metric.item() if isinstance(metric, np.float64) else metric for metric in metric_tuple]
        result_row = list(config) + metric_list
        if include_constant_params_in_output: result_row += list(constant_params.values())
        result_array.append(result_row)  # Append to result array

    # Convert to frame and sort eventually:
    results = pd.DataFrame(result_array, columns=columns)
    if sort_metric is not None:
        minimize = sort_metric in ['StdDev']
        results = results.sort_values(by=sort_metric, ascending=minimize)

    # update backtest_database_dir:
    if backtest_database_dir is not None:
        results['date'] = datetime.today().strftime('%Y-%m-%d')

        try:  # concatenation
            previous_frame = pd.read_csv(filemgmt.most_recent_file(backtest_database_dir))
            new_frame = pd.concat([previous_frame, results], ignore_index=True)
        except ValueError:  # otherwise just save result
            new_frame = results

        try:  # duplicate removal
            new_frame.drop_duplicates(inplace=True)
        except TypeError as err:
            print("Error during duplicate removal:", err)

        # save:
        new_frame.to_csv(backtest_database_dir / filemgmt.file_title("Backtest Database", ".csv"), index=False)

    return results
