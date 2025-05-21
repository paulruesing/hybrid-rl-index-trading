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
                               price_column: Literal["1. open", "2. high", "3. low", "4. close"] = '4. close',
                               csv_path_to_concat: str = None,
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
    :param csv_path_to_concat: str, optional
        Path to csv file from previous download which should be extended.
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
    if csv_path_to_concat is not None:
        try:  # if csv_path_to_concat is a directory:
            csv_path_to_concat = filemgmt.most_recent_file(csv_path_to_concat, ".csv", ticker)
            print(f'Since provided csv_path_to_concat is a directory will now load: {csv_path_to_concat}')
        except NotADirectoryError:
            pass  # in that case no change necessary

        # load existing dataframe:
        price_frame = pd.read_csv(csv_path_to_concat)
        # set datetime as index:
        price_frame['date'] = pd.to_datetime(price_frame['date'])
        price_frame.set_index('date', inplace=True)
    else:
        price_frame = pd.DataFrame()

    # query and concat:
    for year_month in tqdm(list_of_year_month_strings):
        ts = TimeSeries(key=api_key, output_format='pandas')  # initialise time-series API
        try:  # query:
            temp_price_frame = \
            ts.get_intraday(ticker, extended_hours=False, interval=sampling_rate, month=year_month, outputsize="full")[
                0]
            price_frame = pd.concat([price_frame, temp_price_frame])
        except ValueError as err:  # occurs if capacity for free queries is exhausted
            print(err)
    # sort the data according to datetimes and remove duplicates:
    price_frame.sort_index(inplace=True)
    price_frame.drop_duplicates(inplace=True)

    # save data:
    if save_path is not None:
        date_range_string = f"{price_frame.index.min().strftime('%Y-%m-%d')} to {price_frame.index.max().strftime('%Y-%m-%d')}"
        save_title = filemgmt.file_title(title=f"{ticker} {price_column} price data {date_range_string}",
                                         dtype_suffix=".csv")
        price_frame.to_csv(save_path / save_title)
        return price_frame

    return price_frame


def time_interpolation_new_sampling_rate(df: Union[pd.DataFrame, pd.Series], interpolation_column: str,
                                         datetime_column: str,
                                         new_sampling_rate: str = '1min',
                                         custom_start_hour: int = None, custom_start_minute: int = None,
                                         df_lowest_time_unit: Literal['1min', '1sec'] = '1min',
                                         verbose=False,
                                         save_path=None, save_title_identifier: str = None,
                                         new_price_column_label: str = 'close',
                                         exclude_non_operating_hours=True, manual_operating_hours: (int, int) = None,
                                         exclude_weekends=True):
    """
    Interpolate samples of a dataframe to fit a new (higher) sampling rate.

    Allows to specify start point of interpolated series besides imported data starting point through
    custom_start_hour and custom_start_minute.

    Respects weekends (if exclude_weekends) and operating hours (if exclude_non_operating_hours, also manually possible with manual_operating_hours e.g. = (9, 18)).
    """
    # prepare data
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, [datetime_column, interpolation_column]].copy()
    else:
        df = pd.DataFrame(df.copy()).reset_index()
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

    # renaming:
    interpolated_prices.rename(columns={interpolation_column: new_price_column_label}, inplace=True)
    interpolated_prices.index.name = datetime_column

    # save data:
    if save_path is not None:
        date_range_string = f"{interpolated_prices.index.min().strftime('%Y-%m-%d')} to {interpolated_prices.index.max().strftime('%Y-%m-%d')}"
        save_title = filemgmt.file_title(
            title=f"{f' {save_title_identifier} ' if save_title_identifier is not None else ''}Interpolated Prices at {new_sampling_rate} from {date_range_string}",
            dtype_suffix=".csv")
        interpolated_prices.to_csv(save_path / save_title)

    return interpolated_prices