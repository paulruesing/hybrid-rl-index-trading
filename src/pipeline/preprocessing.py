import os
from datetime import datetime

import yfinance as yf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Union
from tqdm import tqdm
from alpha_vantage.timeseries import TimeSeries

import src.utils.file_management as filemgmt
import src.pipeline.web_interaction as webinteraction


class Normaliser():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        """ Normalise an array of values. """
        self.mu = np.mean(x, axis=0)
        self.sd = np.std(x, axis=0)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def transform(self, x):
        """ Normalise additional data of same sequence as before. """
        if self.sd is None: raise AttributeError(
            "Please use fit_transform first so this instance remembers the respective std. and mean values!")
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        """ Reverse-transform an array of normalised values. """
        if self.sd is None: raise AttributeError(
            "Please use fit_transform first so this instance remembers the respective std. and mean values!")
        return (x * self.sd) + self.mu


class StockPriceDataManager:
    """
    Class for managing stock price data, including downloading, interpolating, and adjusting ETF data.

    Attributes
    ----------
    ticker_symbol : str
        The ticker symbol for the stock or ETF.
    download_dir : Path
        Directory where downloaded data files are stored.
    interpolated_files_dir : Path
        Directory where interpolated data files are stored.
    env_sampling_rate_minutes : Literal[15, 60, 1440]
        Sampling rate in minutes for interpolated data.
    price_column : Literal['low', 'high', 'open', 'close']
        Price column to use for calculations and interpolations.
    is_etf_price_data : bool
        Indicates if the data pertains to ETF prices.
    non_etf_time_price_tuples : [(pd.Timestamp, float)]
        Time and price tuples for non-ETF counterpart data.
    download_time_increment : str
        Time increment for scheduling downloads, e.g., "6h".
    manual_operating_h_tuple : (int, int)
        Tuple representing manual operating hours (start, end).
    custom_start_h_min_tuple : (int, int)
        Tuple representing custom operating start time (hour, minute).

    Properties
    ----------
    non_etf_time_price_tuples : tuple
        Time and price tuples for non-ETF counterpart data.
    non_etf_price_factor : float
        Factor for converting ETF price data to non-ETF price data.
    env_sampling_rate_str : str
        String representation of the environment sampling rate.
    a_sampling_rate_str : str
        String representation of a-predictor sampling rate.
    b_sampling_rate_str : str
        String representation of b-predictor sampling rate.
    c_sampling_rate_str : str
        String representation of c-predictor sampling rate.
    d_sampling_rate_str : str
        String representation of d-predictor sampling rate.
    env_interp_prices : pd.Series
        Interpolated prices for the environment sampling rate.
    non_etf_env_interp_prices : pd.Series
        Interpolated prices for the environment sampling rate, multiplied by the non-ETF price factor.
    a_interp_prices : pd.Series
        Interpolated prices for the a-predictor sampling rate.
    b_interp_prices : pd.Series
        Interpolated prices for the b-predictor sampling rate.
    c_interp_prices : pd.Series
        Interpolated prices for the c-predictor sampling rate.
    d_interp_prices : pd.Series
        Interpolated prices for the d-predictor sampling rate.
    downloaded_price_frame : pd.DataFrame
        Dataframe of the most recently downloaded price data.
    downloaded_prices : pd.Series
        Series of the most recently downloaded prices according to the specified price column.
    interp_data_time_range : (pd.Timestamp, pd.Timestamp)
        Time range of the interpolated data.
    downloaded_data_time_range : (pd.Timestamp, pd.Timestamp)
        Time range of the downloaded data.

    Methods
    -------
    download_new_data()
        Downloads the most recent price data from AlphaVantage.
    """

    def __init__(self,
                 download_dir: Path,
                 interpolated_files_dir: Path,
                 alpha_vantage_api_key: str,

                 ticker_symbol: str = 'DAX',
                 scrape_url_comdirect: str = "https://www.comdirect.de/inf/fonds/detail/chart.html?ID_NOTATION=115802659&",
                 include_scraped_downloads: bool = False,  # if True -> adds today's scraped data to downloaded_prices
                 scrape_raw_download_dir: Path = None,

                 env_sampling_rate_minutes: Literal[15, 60, 1440] = 15,
                 price_column: Literal['low', 'high', 'open', 'close'] = 'close',
                 is_etf_price_data: bool = False,
                 non_etf_time_price_tuples: [(pd.Timestamp, float)] = None,
                 download_time_increment: str = "6h",  # for US downloads
                 manual_operating_h_tuple: (int, int) = (8, 22),
                 custom_start_h_min_tuple: (int, int) = (16, 0),
                 ):
        self.ticker_symbol = ticker_symbol
        self.scrape_url_comdirect = scrape_url_comdirect
        self.include_scraped_downloads = include_scraped_downloads
        if include_scraped_downloads:
            if scrape_raw_download_dir is None:
                raise ValueError("If include_scraped_downloads is True, scrape_raw_download_dir must be provided.")
        self.scrape_raw_download_dir = scrape_raw_download_dir

        self.download_dir = download_dir
        self.interpolated_files_dir = interpolated_files_dir
        self._alpha_vantage_api_key = alpha_vantage_api_key
        self.env_sampling_rate_minutes = env_sampling_rate_minutes
        self.price_column = price_column
        self.is_etf_price_data = is_etf_price_data
        self._non_etf_time_price_tuples = non_etf_time_price_tuples
        self.download_time_increment = download_time_increment
        self.manual_operating_h_tuple = manual_operating_h_tuple
        self.custom_start_h_min_tuple = custom_start_h_min_tuple

        # placeholders for properties:
        self._non_etf_price_factor = None

        # not necesssary, reading out is enough:
        self._present_data_timerange = None

    #### Amenable Properties ####
    @property
    def non_etf_time_price_tuples(self) -> tuple:
        return self._non_etf_time_price_tuples

    @non_etf_time_price_tuples.setter
    def non_etf_time_price_tuples(self, value: tuple) -> None:
        self._non_etf_time_price_tuples = value
        self._non_etf_price_factor = None

    #### Saved Properties ####
    @property
    def non_etf_price_factor(self) -> float:
        """
        Fetches or calculates the non-ETF price factor for converting ETF price data into its underlying equivalent.

        Returns
        -------
        float
            A factor to adjust ETF price data, defaulting to 1.0 for non-ETF data. Raises an exception if the required data for computation is missing.

        Raises
        ------
        AttributeError
            If the non-ETF time-price tuples are not provided while ETF price data conversion is needed.
        """
        if not self.is_etf_price_data: return 1.0

        if self._non_etf_price_factor is None:
            if self._non_etf_time_price_tuples is None:
                raise AttributeError(
                    "Non-etf time-price tuples need to be provided for conversion of ETF data into underlying.")
            # compute multipliers and take mean:
            multipliers = [price / self.env_interp_prices.loc[date] for date, price in self.non_etf_time_price_tuples]
            self._non_etf_price_factor = np.mean(multipliers)
        return self._non_etf_price_factor

    ### Read-out at Runtime Properties ####
    ## sampling rates:
    @property
    def env_sampling_rate_str(self) -> str:
        """ Timedelta string related to environment sampling rate based on self.env_sampling_rate_minutes """
        minutes_to_str_dict = {15: self.a_sampling_rate_str, 60: self.b_sampling_rate_str,
                               1440: self.c_sampling_rate_str}
        return minutes_to_str_dict[self.env_sampling_rate_minutes]

    @property  # todo: read out global variable?
    def a_sampling_rate_str(self) -> str:
        return "15min"

    @property  # todo: read out global variable?
    def b_sampling_rate_str(self) -> str:
        return "60min"

    @property  # todo: read out global variable?
    def c_sampling_rate_str(self) -> str:
        return "1d"

    @property  # todo: read out global variable?
    def d_sampling_rate_str(self) -> str:
        return "7d"

    ## price frames:
    @property
    def env_interp_prices(self) -> pd.Series:
        """ Interpolated prices sampled at env_sampling_rate_minutes. """
        return read_price_csv(
            filemgmt.most_recent_file(self.interpolated_files_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol, self.env_sampling_rate_str]),
            date_column='date', price_column=self.price_column)

    @property
    def non_etf_env_interp_prices(self) -> pd.Series:
        """ Interpolated prices sampled at env_sampling_rate_minutes and multiplied by non_etf_price_factor. """
        return read_price_csv(
            filemgmt.most_recent_file(self.interpolated_files_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol, self.env_sampling_rate_str]),
            date_column='date',
            price_column=self.price_column) * self.non_etf_price_factor  # multiply with non-etf-factor

    @property
    def a_interp_prices(self) -> pd.Series:
        """ Interpolated prices sampled at a-predictor-type sampling rate. """
        return read_price_csv(
            filemgmt.most_recent_file(self.interpolated_files_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol, self.a_sampling_rate_str]),
            date_column='date', price_column=self.price_column)

    @property
    def b_interp_prices(self) -> pd.Series:
        """ Interpolated prices sampled at b-predictor-type sampling rate. """
        return read_price_csv(
            filemgmt.most_recent_file(self.interpolated_files_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol, self.env_sampling_rate_str]),
            date_column='date', price_column=self.price_column)

    @property
    def c_interp_prices(self) -> pd.Series:
        """ Interpolated prices sampled at c-predictor-type sampling rate. """
        return read_price_csv(
            filemgmt.most_recent_file(self.interpolated_files_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol, self.c_sampling_rate_str]),
            date_column='date', price_column=self.price_column)

    @property
    def d_interp_prices(self) -> pd.Series:
        """ Interpolated prices sampled at d-predictor-type sampling rate. """
        return read_price_csv(
            filemgmt.most_recent_file(self.interpolated_files_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol, self.d_sampling_rate_str]),
            date_column='date', price_column=self.price_column)

    @property
    def downloaded_price_frame(self) -> pd.DataFrame:
        """ Most recent price download dataframe. """
        frame = pd.read_csv(
            filemgmt.most_recent_file(self.download_dir,
                                      suffix_to_consider=".csv",
                                      file_title_keywords=[self.ticker_symbol])
        )
        frame['date'] = pd.to_datetime(frame['date'])
        return frame.set_index('date').dropna(axis=0)

    @property
    def downloaded_prices(self) -> pd.Series:
        """
        Most recent price download series according to self.price_column.
        Includes scraped prices if self.include_scraped_prices is True.
        """
        alpha_vantage_column_renaming = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'}
        price_frame = self.downloaded_price_frame.rename(columns=alpha_vantage_column_renaming)
        price_series = price_frame[self.price_column]

        if self.include_scraped_downloads:
            scrape_series = webinteraction.fetch_price_from_comdirect(raw_download_dir=self.scrape_raw_download_dir,
                                                                      url=self.scrape_url_comdirect,
                                                                      )
            # format scraped series:
            scrape_series.name = self.price_column; scrape_series.index.name = 'date'

            # concatenate:
            price_series = pd.concat([price_series, scrape_series])
            price_series = price_series[~price_series.index.duplicated(keep='first')]  # drop duplicate indices

        return price_series

    ## other:
    @property
    def interp_data_time_range(self) -> (pd.Timestamp, pd.Timestamp):
        """ Time range of interpolated data. """
        date_times = self.env_interp_prices.index  # we assume all time ranges are equal because all are updated jointly
        return date_times.min(), date_times.max()

    @property
    def downloaded_data_time_range(self) -> (pd.Timestamp, pd.Timestamp):
        """ Time range of downloaded data. """
        date_times = self.downloaded_prices.index  # we assume all time ranges are equal because all are updated jointly
        return date_times.min(), date_times.max()

    #### Methods ####
    def download_new_data(self):
        """ Download most recent data from AlphaVantage """
        # define timerange to download:
        latest_data = self.downloaded_data_time_range[1]
        start_tuple = (latest_data.year, latest_data.month)
        end_tuple = (datetime.today().year, datetime.today().month)

        get_data_from_alphavantage(self._alpha_vantage_api_key,
                                   ticker=self.ticker_symbol,
                                   save_path=self.download_dir,
                                   start_year_month=start_tuple,
                                   end_year_month=end_tuple,
                                   price_frame_to_concat=self.downloaded_price_frame,
                                   time_increment=self.download_time_increment)

    def update_interpolated_data(self):
        """ Interpolate and sample all price files based on most recent data download. """
        raw_price_series = self.downloaded_prices
        for sampling_rate_str in [self.a_sampling_rate_str, self.b_sampling_rate_str, self.c_sampling_rate_str,
                                  self.d_sampling_rate_str]:
            time_interpolation_new_sampling_rate(price_series=raw_price_series,
                                                 datetime_column='date',
                                                 new_sampling_rate=sampling_rate_str,
                                                 moving_average_window_size=None,
                                                 custom_start_hour=
                                                 self.custom_start_h_min_tuple[0],
                                                 custom_start_minute=
                                                 self.custom_start_h_min_tuple[1],
                                                 verbose=False,
                                                 manual_operating_hours=self.manual_operating_h_tuple,
                                                 save_path=self.interpolated_files_dir,
                                                 save_title_identifier=f'{self.ticker_symbol} {self.price_column}')

    def update(self):
        """ Update downloaded and interpolated data. """
        print(
            f"Downloaded data ranged from\t{self.downloaded_data_time_range[0]} to {self.downloaded_data_time_range[1]}")
        self.download_new_data()
        print(f"now ranges from\t\t\t\t{self.downloaded_data_time_range[0]} to {self.downloaded_data_time_range[1]}")
        print(f"\nInterpolated data ranged from\t{self.interp_data_time_range[0]} to {self.interp_data_time_range[1]}")
        self.update_interpolated_data()
        print(f"now ranges from\t\t\t\t\t{self.interp_data_time_range[0]} to {self.interp_data_time_range[1]}")


### Data manipulation functions
def time_interpolation_new_sampling_rate(price_series: pd.Series,
                                         datetime_column: str = 'date',
                                         new_sampling_rate: str = '1min',
                                         moving_average_window_size: str = None,
                                         custom_start_hour: int = None, custom_start_minute: int = None,
                                         df_lowest_time_unit: Literal['15min', '1min', '1sec'] = '1min',
                                         verbose=False,
                                         save_path=None, save_title_identifier: str = None,
                                         new_price_column_label: str = None,
                                         exclude_non_operating_hours=True, manual_operating_hours: (int, int) = None,
                                         exclude_weekends=True):
    """
    Interpolate a time series to a new (higher) sampling rate, with optional smoothing and filtering.

    This function resamples a time-indexed DataFrame or Series to a finer-grained time resolution
    using time-based interpolation. Users can customize the resampling behavior through parameters
    like custom start time, moving average smoothing, weekend exclusion, operating hours filtering,
    and export of the resulting DataFrame.

    Parameters
    ----------
    df : DataFrame or Series
        Time series data containing a datetime column and the column to interpolate.
    interpolation_column : str
        Name of the column containing the values to interpolate.
    datetime_column : str
        Name of the datetime column used to index and resample the data.
    new_sampling_rate : str, default '1min'
        Target sampling frequency (e.g., '1min', '30s', '5min') in pandas frequency string format.
    moving_average_window_size : str, optional
        Window size for optional moving average smoothing after interpolation (e.g., '3min').
    custom_start_hour : int, optional
        Custom hour at which the interpolated time series should begin (overrides original start hour).
    custom_start_minute : int, optional
        Custom minute at which the interpolated time series should begin (overrides original start minute).
    df_lowest_time_unit : {'15min', '1min', '1sec'}, default '1min'
        Granularity of the original data; used to determine if outer join is needed during interpolation.
    verbose : bool, default False
        If True, prints detailed information about the interpolation and filtering steps.
    save_path : path-like, optional
        Directory path where the interpolated result will be saved as a CSV, if specified.
    save_title_identifier : str, optional
        Additional identifier for the exported file name.
    new_price_column_label : str, default 'close'
        Name to assign to the interpolated value column in the output.
    exclude_non_operating_hours : bool, default True
        Whether to exclude values outside the operating hours.
    manual_operating_hours : tuple of (int, int), optional
        Tuple specifying manual operating hours (start_hour, end_hour). Used if `exclude_non_operating_hours=True`.
    exclude_weekends : bool, default True
        Whether to exclude data points falling on Saturday or Sunday.

    Returns
    -------
    DataFrame
        Interpolated time series with optional smoothing and filtering applied. The index is datetime.

    Notes
    -----
    - Time-based interpolation (`method='time'`) is used to account for uneven original timestamps.
    - Weekend and operating hour exclusion is applied after index generation and before interpolation.
    - If the new sampling rate is finer than the original data's granularity, an outer join is performed
      to preserve original data before interpolation.
    - Use `moving_average_window_size` to apply rolling mean smoothing after interpolation.

    Examples
    --------
    >>> df = pd.DataFrame({'timestamp': [...], 'price': [...]})
    >>> result = time_interpolation_new_sampling_rate(df, interpolation_column='price',
    ...                                               datetime_column='timestamp',
    ...                                               new_sampling_rate='1min',
    ...                                               exclude_weekends=True,
    ...                                               manual_operating_hours=(9, 17))
    """
    # prepare data
    df = pd.DataFrame(price_series.copy()).reset_index()  # create dataframe
    interpolation_column = price_series.name
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df.set_index(datetime_column, inplace=True)

    # create optimal datetime index:
    date_start = df.index.min()
    # eventually change starting hour and minute, if such arguments are None, replace yields the unchanged timestamp
    date_start = date_start.replace(hour=custom_start_hour, minute=custom_start_minute)
    optimal_date_range = pd.date_range(date_start, df.index.max(), freq=new_sampling_rate)
    if exclude_weekends:  # exclude weekends
        optimal_date_range = optimal_date_range[~optimal_date_range.weekday.isin([5, 6])]
        if verbose:  print("Excluded every entry on Saturday or Sunday.")
    if exclude_non_operating_hours:  # exclude non-operating hours
        operating_hours = (
            df.index.hour.min(), df.index.hour.max()) if manual_operating_hours is None else manual_operating_hours
        optimal_date_range = optimal_date_range[
            ~ ((optimal_date_range.hour >= operating_hours[1]) | (optimal_date_range.hour < operating_hours[0]))]
        if verbose: print(f"Excluded every entry before hour {operating_hours[0]} and after hour {operating_hours[1]}.")

    # below we will join the new datetime indices with the existing ones:
    #   how='left' (default) leads to keeping only new indices, if sampling_rate_to_be_interpolated is too low, information is lost
    #   how='outer' can mitigate this, however then different sample rates remain: might be reasonable for interpolation but should then be removed again
    # we check for this necessity with the outer_join_necessary bool:
    outer_join_necessary = (new_sampling_rate != df_lowest_time_unit)
    if verbose and outer_join_necessary:
        print(
            f"New sampling rate ({new_sampling_rate}) is higher than current lowest time unit ({df_lowest_time_unit}).\nTherefore some existing indices will be removed to match the new sampling rate, while all information will be kept through the interpolation procedure.")
    how_to_join = 'outer' if outer_join_necessary else 'left'
    interpolated_prices = pd.DataFrame(data=None,
                                       index=optimal_date_range)  # new dataframe as scaffold for future interpolated prices
    interpolated_prices = interpolated_prices.join(df[interpolation_column], how=how_to_join)  # join prices

    # interpolation:
    #   we use 'time' interpolation because we have unevenly spaced time indices in the original time-series
    #   if time-indices are equally spaced, 'time' becomes equivalent to 'linear' interpolation
    interpolated_prices = interpolated_prices.interpolate(method='time')

    # if we used 'outer' join, we now need to remove the unequally spaced indices:
    if outer_join_necessary:
        interpolated_prices = interpolated_prices.loc[interpolated_prices.index.isin(optimal_date_range)]

    # eventually smooth with moving average:
    if moving_average_window_size is not None:
        interpolated_prices = interpolated_prices.rolling(window=moving_average_window_size).mean()
        if verbose:
            print(f"Smoothed interpolated data using moving average with {moving_average_window_size} window.")

    # renaming:
    if new_price_column_label is not None:
        interpolated_prices.rename(columns={interpolation_column: new_price_column_label}, inplace=True)
    interpolated_prices.index.name = datetime_column

    # save data:
    if save_path is not None:
        date_range_string = f"{interpolated_prices.index.min().strftime('%Y-%m-%d')} to {interpolated_prices.index.max().strftime('%Y-%m-%d')}"
        save_title = filemgmt.file_title(
            title=f"{f' {save_title_identifier} ' if save_title_identifier is not None else ''}Interpolated Prices at {new_sampling_rate} {f'smoothed over {moving_average_window_size} ' if moving_average_window_size is not None else ''}from {date_range_string}",
            dtype_suffix=".csv")
        interpolated_prices.to_csv(save_path / save_title)

    return interpolated_prices


def create_rolling_window_view(input_series: pd.Series,
                               rolling_window_size: int, forecast_horizon: int = 1,
                               daily_prediction_hour: int = None, sampling_rate_minutes: int = 15,
                               predict_before_daily_prediction_hour: bool = False,
                               verbose=False):
    """
    Creates a rolling window matrix of training data and target values based on a time-series with datetime index.
    Uses subsequent prices as target values, i.e. autoregressive prediction.

    Columns of training data are defined by rolling_window_size, columns of target data by forecast_horizon.

    For intra-day predictions, the method allows for defining the ending point of each rolling window (and
    hence starting point of target values) with daily_prediction_hour. E.g. daily_prediction_hour=15 will lead
    to the targets always starting at the first sample after 3 pm.
    This further requires specifying sampling_rate_minutes to find the first entry in that prediction hour.
    """
    ### create rolling window views:
    # sliding window view as matrix: last column are current prices, 1st to (rolling-window-size - 1)th column are retrospective prices:
    X = np.lib.stride_tricks.sliding_window_view(input_series.to_numpy(),
                                                       window_shape=rolling_window_size)[
              :-forecast_horizon]  # last rows (latest values) are removed (because contained only in target values)
    X_dates = np.lib.stride_tricks.sliding_window_view(input_series.index.to_numpy(),
                                                             window_shape=rolling_window_size)[:-forecast_horizon]

    # target values are subsequent prices, window size here is referred to as the forecast_horizon:
    Y = np.lib.stride_tricks.sliding_window_view(input_series.to_numpy(),
                                                       window_shape=forecast_horizon)[
              rolling_window_size:]  # first rows (earliest values) are removed (because contained only in training values)
    Y_dates = np.lib.stride_tricks.sliding_window_view(input_series.index.to_numpy(),
                                                             window_shape=forecast_horizon)[
                    rolling_window_size:]  # first rows (earliest values) are removed (because contained only in training values)
    if verbose: print(
        f"Created rolling window view based on rolling_window_size of {rolling_window_size} and forecast_horizon of {forecast_horizon} with a time unit of {sampling_rate_minutes} minutes.")

    if daily_prediction_hour is not None:
        # specify prediction start mask:
        clipped_input_series = input_series.iloc[rolling_window_size:]  # first rows are removed because necessary for rolling window
        if forecast_horizon > 1:  # last rows are removed because necessary for targets that exceed provided data
            clipped_input_series = clipped_input_series.iloc[:-forecast_horizon + 1]
        target_date_index = clipped_input_series.index  # relevant dates
        if sampling_rate_minutes >= 60:  # if sampling rate larger than 1 hour, no need for minute check:
            prediction_start_mask = (target_date_index.hour == daily_prediction_hour)
            if verbose: print(
                f"Target values start at only observation between {daily_prediction_hour}:00 and {daily_prediction_hour+1}:00 daily.")
        else:  # check also for minute of observation:
            if predict_before_daily_prediction_hour:  # predict at last observation before prediction hour
                prediction_start_mask = (target_date_index.hour == daily_prediction_hour - 1) & (
                        target_date_index.minute >= (
                            60 - sampling_rate_minutes))  # last observation before prediction_hour
                if verbose: print(
                    f"Target values start at last observation before {daily_prediction_hour}:00 daily.")
            else:  # predict at first observation in prediction_hour
                prediction_start_mask = (target_date_index.hour == daily_prediction_hour) & (
                        target_date_index.minute < sampling_rate_minutes)  # first observation in prediction_hour
                if verbose: print(
                    f"Target values start at first observation after {daily_prediction_hour}:00 daily.")

        # select only training values related to target values starting at the specified prediction time:
        X = X[prediction_start_mask]
        Y = Y[prediction_start_mask]
        X_dates = X_dates[prediction_start_mask]
        Y_dates = Y_dates[prediction_start_mask]

        # status message and sanity check:
        if len(X) > 0:
            if verbose: print(f"Resulting dataset consists of {len(X)} observations.")
        else:
            raise ValueError(f"No observations remain after choosing observations according to prediction hour of {daily_prediction_hour}.\nThis can be due to wrong specification of the sampling rate (currently {sampling_rate_minutes} min)!")

    return X, Y, X_dates, Y_dates


def create_train_validation_split(X: np.ndarray, Y: np.ndarray,
                                  X_dates: np.ndarray, Y_dates: np.ndarray,
                                  validation_split: float = 0.2,
                                  randomise: bool = True,
                                  verbose: bool = False):
    """
    Splits training and target values into training and validation split.

    Returns tuple with X_train, X_val, Y_train, Y_val, X_dates_train, X_dates_val, Y_dates_train, Y_dates_val.
    """
    if randomise:
        # shuffle row indices:
        idx = np.random.permutation(X.shape[0])

        # apply to all matrices so that rows that belong together, still have the same index:
        X = X[idx]; X_dates = X_dates[idx]; Y = Y[idx]; Y_dates = Y_dates[idx]
        # afterwards the same procedure as without randomisation can be kept

    # derive index separating last rows of data:
    validation_split_index = int(X.shape[0] * (1 - validation_split))

    # split train and validation values:
    X_train = X[:validation_split_index]; X_val = X[validation_split_index:]
    Y_train = Y[:validation_split_index]; Y_val = Y[validation_split_index:]
    if verbose:
        print(f"Using last {100 * validation_split}% of data for validation. Other data for training.")
        print(f"This yields {len(X_train)} training and {len(X_val)} validation observations.")

    # split respective dates:
    X_dates_train = X_dates[:validation_split_index]; X_dates_val = X_dates[validation_split_index:]
    Y_dates_train = Y_dates[:validation_split_index]; Y_dates_val = Y_dates[validation_split_index:]

    return X_train, X_val, Y_train, Y_val, X_dates_train, X_dates_val, Y_dates_train, Y_dates_val





### Data download functions
def read_price_csv(csv_path: str, date_column: str = "date", price_column: str = "close") -> pd.Series:
    """ Read price csv file. Type conversion, NA-cleaning and formatting. """
    # import csv w correct dtypes:
    price_file = pd.read_csv(csv_path).dropna(axis=0)
    try:
        price_file[date_column] = pd.to_datetime(price_file[date_column])
    except KeyError:  # if the csv has no name for its index:
        price_file[date_column] = pd.to_datetime(price_file['Unnamed: 0'])
    price_file[price_column] = price_file[price_column].astype(float)
    return price_file.set_index(date_column)[price_column]


def get_data_from_yahoo(ticker: str = '^GDAXI', duration_days: int = None, sampling_rate_minutes: int = None,
                        sampling_rate_days: int = 1, verbose=True, m_avg_days=[5, 30, 90],
                        price_column='Close', validation_split: float = None,
                        start_date=None, end_date=None, save_path=None) -> pd.DataFrame:
    """
    Data downloader utilising yfinance library.

    :param ticker: string, required, default = "^GDAXI" (DAX index)
        Specifies which stock's price data to download. Examples: "^IXIC" (Nasdaq), "^DJI" (Dow-Jones), "^GSPC" (S&P 500)
    :param duration_days: int, optional
        Amount of days to download data for. If not specified, downloads as much as possible.
    :param sampling_rate_minutes: int, optional
        Sampling rate of data in minutes. If not provided, utilises sampling_rate_days.
    :param sampling_rate_days: int, optional, default = 1
        Sampling rate of data in days.
    :param verbose: Bool, optional, default = True
        Specifies whether to print status messages e.g. regarding data coverage.
    :param m_avg_days: list of ints, optional, default = [5, 30, 90]
        Days to calculate moving averages for. If set to None, will calculate no moving averages.
    :param price_column: str, optional, default = "Close"
        Which of the downloaded columns to utilise as price data. Options: "Close", "High", "Low", "Open"
    :param validation_split: float, optional
        If provided, will return tuple with (training-split, validation-split) and the latter containing the respective ratio of latest data.
    :param start_date:
    :param end_date:
    :param save_path: str, optional, default = None
        File-title string defining repository where to save the downloaded data. If not provided, will not save the results.
    :return:
    """
    # download data:
    period = f'{duration_days}d' if duration_days is not None else 'max'
    interval = f'{sampling_rate_minutes}m' if sampling_rate_days is None else f'{sampling_rate_days}d'
    data = yf.download(tickers=ticker, period=period, interval=interval,
                       start=start_date, end=end_date, progress=verbose)

    # calculate data coverage subtracting weekends:
    if verbose:
        pandas_freq = f'{sampling_rate_days}D' if sampling_rate_days is not None else f'{sampling_rate_minutes}min'
        compare_dt_index = pd.date_range(start=data.index[0], end=data.index[-1], freq=pandas_freq)
        compare_dt_index = compare_dt_index[~compare_dt_index.weekday.isin([5, 6])]  # exclude weekends
        compare_dt_index = compare_dt_index[~ ((compare_dt_index.hour < data.index.hour.min()) | (
                compare_dt_index.hour > data.index.hour.max()))]  # exclude non-operating hours
        print(
            f"Coverage of downloaded data: {round(len(data) / len(compare_dt_index) * 100, 1)}%! Weekends and non-operating hours excluded. Can be due to holidays or missing data.")
        data[price_column].plot()

    # eventually calculate moving averages:
    if m_avg_days is not None:
        for days in m_avg_days:
            data[f"MA{days}"] = data[price_column].rolling(window=f"{days}d").mean()

    # remove index:
    data.reset_index(inplace=True, names="Datetime")

    # save data:
    if save_path is not None:
        date_range_string = f"{data.Datetime.min().strftime('%Y-%m-%d')} to {data.Datetime.max().strftime('%Y-%m-%d')}"
        save_title = filemgmt.file_title(title=f"{ticker} {price_column} price data {date_range_string}",
                                         dtype_suffix=".csv")
        data.to_csv(save_path / save_title)

    # return with or without validation split:
    if validation_split is not None:
        if verbose: print(f"Returning last {100 * validation_split}% of data for validation. Other data for training.")
        validation_split_index = int(len(data) * (1 - validation_split))
        return (data.iloc[:validation_split_index], data.iloc[validation_split_index:])
    else:
        return data


def get_data_from_alphavantage(api_key: str,
                               ticker: str = 'DAX',
                               start_year_month: (int, int) = None, end_year_month: (int, int) = None,
                               sampling_rate: Literal['1min', '5min', '15min', '30min', '60min'] = '1min',
                               time_increment: str = None, time_decrement: str = None,
                               price_csv_dir_to_concat: Union[Path, str] = None,
                               price_frame_to_concat: pd.DataFrame = None,
                               save_path=None) -> pd.DataFrame:
    """
    Data downloader utilising alpha-vantage's API.

    :param ticker: string, default = "DAX" (DAX index)
        Specifies which stock's price data to download. Examples: "VCNIX" (Nasdaq)
    :param start_year_month: tuple (int, int), optional
        Amount of days to download data for. If not specified, downloads as much as possible.
    :param end_year_month: tuple (int, int), optional
        Amount of days to download data for. If not specified, downloads as much as possible.
    :param sampling_rate: str, default = "1min"
        Sampling rate of data. Options: '1min', '5min', '15min', '30min', '60min'
    :param price_column: str, default = "4. close"
        Which of the downloaded columns to utilise as price data. Options: "1. open", "2. high", "3. low", "4. close"
    :param time_increment: str, optional
        Time-zone adjustment. Will be added to date column. E.g. '6h' for adjusting from UCT-4 (New York) to UCT+2 (Frankfurt)
    :param time_decrement: str, optional
        Time-zone adjustment. Will be subtracted from date column.
    :param price_csv_dir_to_concat: str, optional
        Dir of csv files from previous download, the most recent file will be extended.
    :param price_frame_to_concat: pd.DataFrame, optional
        Dataframe from previous download to be extended. Overrules price_csv_dir_to_concat.
    :param save_path: str, optional
        File-title string defining repository where to save the downloaded data. If not provided, will not save the results.

    :return:
    """
    # if no end date specified download until start date:
    if start_year_month is not None and end_year_month is None:
        end_year_month = start_year_month

    # prepare month arguments for API queries:
    list_of_year_month_strings = []
    if start_year_month is not None:
        for year in range(start_year_month[0], end_year_month[0] + 1):
            if year == start_year_month[0] and year == end_year_month[0]:  # if only one year
                for month in range(start_year_month[1], end_year_month[1] + 1):
                    list_of_year_month_strings.append(f"{year}-{f'{month}' if month >= 10 else f'0{month}'}")
            elif year == start_year_month[0]:  # first year
                for month in range(start_year_month[1], 13):
                    list_of_year_month_strings.append(f"{year}-{f'{month}' if month >= 10 else f'0{month}'}")
            elif year == end_year_month[0]:  # last year
                for month in range(1, end_year_month[1] + 1):
                    list_of_year_month_strings.append(f"{year}-{f'{month}' if month >= 10 else f'0{month}'}")
            else:  # other years
                for month in range(1, 13):
                    list_of_year_month_strings.append(f"{year}-{f'{month}' if month >= 10 else f'0{month}'}")
        if len(list_of_year_month_strings) > 25:
            raise ValueError(
                f"Specified time range would result in more than 25 API queries. This exceeds the daily limit and leads to errors.")
        else:
            print(
                f"Will query the AlphaVantage API {len(list_of_year_month_strings)} times based on the specified time range. Queries:\n",
                list_of_year_month_strings)
    else:
        print("No time range specified. Will download price data of last 30 days.")
        list_of_year_month_strings.append(None)

    # prepare dataframe or load existing one:
    if price_frame_to_concat is not None or price_csv_dir_to_concat is not None:
        if price_frame_to_concat is None:  # load from csv dir
            try:  # if price_csv_dir_to_concat is a directory:
                price_csv_dir_to_concat = filemgmt.most_recent_file(price_csv_dir_to_concat, ".csv", ticker)
                print(f'Since provided price_csv_dir_to_concat is a directory will now load: {price_csv_dir_to_concat}')
            except NotADirectoryError:
                pass  # in that case no change necessary

            # load existing dataframe:
            price_frame = pd.read_csv(price_csv_dir_to_concat)
            # set datetime as index:
            price_frame['date'] = pd.to_datetime(price_frame['date'])
            price_frame.set_index('date', inplace=True)
            
        else:  # utilize provided dataframe
            price_frame = price_frame_to_concat

    else:  # don't concat anything
        price_frame = pd.DataFrame()

    # query and concat:
    for year_month in tqdm(list_of_year_month_strings):
        ts = TimeSeries(key=api_key, output_format='pandas')  # initialise time-series API
        try:  # query:
            temp_price_frame = \
            ts.get_intraday(ticker, extended_hours=False, interval=sampling_rate, month=year_month, outputsize="full")[
                0]

            temp_price_frame.reset_index(inplace=True)
            # eventually adjust for foreign time-zone:
            temp_price_frame['date'] = pd.to_datetime(temp_price_frame['date'])
            if time_increment is not None: temp_price_frame['date'] + pd.Timedelta(time_increment)
            if time_decrement is not None: temp_price_frame['date'] - pd.Timedelta(time_decrement)
            temp_price_frame.set_index('date', inplace=True)
            price_frame = pd.concat([price_frame, temp_price_frame])
        except ValueError as err:  # occurs if capacity for free queries is exhausted
            print(err)
    # sort the data according to datetimes and remove duplicates:
    price_frame.sort_index(inplace=True)
    #price_frame.drop_duplicates(inplace=True)
    price_frame = price_frame[~price_frame.index.duplicated(keep='first')]  # drop duplicate indices

    # save data:
    if save_path is not None:
        date_range_string = f"{price_frame.index.min().strftime('%Y-%m-%d')} to {price_frame.index.max().strftime('%Y-%m-%d')}"
        save_title = filemgmt.file_title(title=f"{ticker} price data {date_range_string}",
                                         dtype_suffix=".csv")
        price_frame.to_csv(save_path / save_title)
        return price_frame

    return price_frame