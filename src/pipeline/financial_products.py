import os
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import src.utils.str_conversion as strconv


class KOCertificate:
    """ Models the price evolution of a future financial product. """
    def __init__(self,
                 underlying_price_series: pd.Series,
                 isin: str = None,
                 risk_premium: float = 0,
                 date_base_price_tuple: (str, float) = None,
                 date_base_price_tuple2: (str, float) = None,
                 abs_base_price_change_per_annum: float = 0.02,  # seems to be reasonable
                 subscription_ratio: float = 1,
                 issue_date: str = None,
                 direction: Literal['long', 'short'] = None,
                 scrape_data_if_possible: bool = True,
                 scrape_driver_executable_path: str = "",
                 ):
        """
        Class for modeling the price evolution of a future financial product
        based on an underlying asset, accounting for base price changes and
        financing costs over time.

        Product information can be automatically scraped by providing an ISIN
        and keeping scrape_data_if_possible at True.

        The class computes a base price series, intrinsic value, and final
        price of a future product, given a time series of underlying prices
        and one or two calibration points. If two base prices or a historical
        future price are provided, it estimates the time-varying base price
        through linear interpolation.

        Parameters
        ----------
        underlying_price_series : pd.Series
            Time series of the underlying asset prices. Must have a DateTimeIndex.
        isin : str
            Can be provided to scrape the product's data risk_premium, subscription_ratio, date_base_price_tuple and issue_date. Will overwrite such.
        risk_premium : float
            Constant premium added to the intrinsic value to compute the product price.
        date_base_price_tuple : (str, float)
            A tuple containing a date and the corresponding base price.
        date_base_price_tuple2 : (str, float), optional
            A second base price tuple (date and value) for interpolation.
        base_price_change_per_annum : float, default is 0.02
            Annual base price increase used for base price estimation if date_base_price_tuple2 not provided.
        subscription_ratio : float, default 1
            Scaling factor that links changes in the underlying asset to the
            product's intrinsic value.
        issue_date : str, optional
            If provided, trims all data before this date in the underlying series.
        direction : "long" or "short", optional
            Fixes product type. If not provided, such property (direction) is inferred from last base and underlying price.
        scrape_data_if_possible : bool, default = True
            If true, product properties will be scraped from boerse-frankfurt.de through the provided ISIN.
        scrape_driver_executable_path : str, optional
            Path to executable driver for Selenium product info scrape. Often not necessary, system can try to automatically locate it.
        """
        # convert underlying price array to pd.Series with DatetimeIndex:
        self._underlying_price_series = underlying_price_series if isinstance(underlying_price_series,
                                                                             pd.Series) else pd.Series(
            underlying_price_series.iloc[:, 0])
        self._underlying_price_series.index = pd.to_datetime(underlying_price_series.index)

        # if provided, overwrite all parameters through web scrape:
        if isin is not None and scrape_data_if_possible:
            risk_premium, subscription_ratio, current_base_price, issue_date = fetch_future_info_from_boerse_fra(isin,
                                                                                                                 scrape_driver_executable_path)
            date_base_price_tuple = (datetime.today().strftime('%Y-%m-%d'), current_base_price)
        self.isin = isin

        self._issue_date = issue_date
        self.risk_premium = risk_premium
        self._date_base_price_tuple = date_base_price_tuple  # will be accessible through properties with setter for recalculation of base-price series
        self._date_base_price_tuple2 = date_base_price_tuple2
        self._base_price_change_per_annum = abs_base_price_change_per_annum
        self.subscription_ratio = subscription_ratio
        self._direction = direction

        # private attributes for properties:
        self._is_ko_series = None
        self._base_price_series = self._intrinsic_value_series = None

    def get_base_price_from_future_price(self, date: str, future_price: float,
                                         use_as_1st_base_price_tuple=False, use_as_2nd_base_price_tuple=False):
        """
        Compute the historic base price from a given past future price.

        Parameters
        ----------
        date : str
            Date of the historical future price.
        future_price : float
            Value of the future product on the given date.
        use_as_1st_base_price_tuple : bool, default False
            If True, sets the computed base price as the first calibration point.
        use_as_2nd_base_price_tuple : bool, default False
            If True, sets the computed base price as the second calibration point.

        Returns
        -------
        float
            The computed historic base price.
        """
        date = pd.Timestamp(date)
        if date.hour == 0: date = date.replace(hour=10)  # prevent errors if no hour was provided

        # computation:
        if self.direction == "long":
            base_price = (self.risk_premium - future_price) / self.subscription_ratio + self.underlying_price_series[date]
        else:  # for short
            base_price = (future_price - self.risk_premium) / self.subscription_ratio + self.underlying_price_series[date]

        # eventually rewrite base price tuples for recalculation of base price series:
        if use_as_1st_base_price_tuple: self.date_base_price_tuple = (date, base_price)
        if use_as_2nd_base_price_tuple: self.date_base_price_tuple2 = (date, base_price)

        return base_price

    def get_base_price_from_leverage(self, date: str, leverage: float,
                                         use_as_1st_base_price_tuple=False, use_as_2nd_base_price_tuple=False):
        """
        Compute the historic base price from a given past future price.

        Parameters
        ----------
        date : str
            Date of the historical future price.
        leverage : float
            Value of the future product on the given date.
        use_as_1st_base_price_tuple : bool, default False
            If True, sets the computed base price as the first calibration point.
        use_as_2nd_base_price_tuple : bool, default False
            If True, sets the computed base price as the second calibration point.

        Returns
        -------
        float
            The computed historic base price.
        """
        date = pd.Timestamp(date)
        if date.hour == 0: date = date.replace(hour=10)  # prevent errors if no hour was provided

        # computation:
        if self.direction == "long":
            base_price = self.underlying_price_series[date] - (self.underlying_price_series[date] * self.subscription_ratio / leverage - self.risk_premium) / self.subscription_ratio
        else:  # for short
            base_price = self.underlying_price_series[date] + (self.underlying_price_series[date] * self.subscription_ratio / leverage - self.risk_premium) / self.subscription_ratio

        # eventually rewrite base price tuples for recalculation of base price series:
        if use_as_1st_base_price_tuple: self.date_base_price_tuple = (date, base_price)
        if use_as_2nd_base_price_tuple: self.date_base_price_tuple2 = (date, base_price)

        return base_price

    def plot(self, plot_size=(10, 10), leverage_lim=(0, 10)) -> None:
        """ Plot price and leverage development. """
        fig, (ax, ax3) = plt.subplots(2, 1, figsize=plot_size)
        ax2 = ax.twinx()
        ax.plot(self.date_index, self.underlying_price_series, color='blue', label='Underlying Price')
        ax.plot(self.date_index, self.base_price_series, color='black', label='Base Price')
        if self.date_base_price_tuple is not None:
            ax.plot(self.date_base_price_tuple[0], self.date_base_price_tuple[1], 'ro', label='Base Price Anchor Point 1')
        if self.date_base_price_tuple2 is not None:
            ax.plot(self.date_base_price_tuple2[0], self.date_base_price_tuple2[1], 'mo', label='Base Price Anchor Point 2')
        ax2.plot(self.date_index, self.price_series, color='green', label='Future Price')
        ax.set_ylabel('Price [€]')
        ax2.set_ylabel('Price [€]')

        # plot leverage:
        ax3.plot(self.date_index, self.leverage_series, color='red', label='Leverage')
        ax.legend(loc='upper left')
        ax2.legend(loc='lower right')
        ax3.set_ylabel('Leverage [x]')
        ax3.set_xlabel('Date')
        ax3.set_ylim(leverage_lim)
        ax.grid(color='grey')
        ax2.grid(axis='y', color='lightgrey')
        ax3.grid()
        fig.tight_layout()

    def fix_initial_knockout(self) -> None:
        """ Reduce base price below minimum of underlying price series to fix initial KO breach. """
        if self.is_ko_series.iloc[0]:
            lowest_date = self.date_index[np.argmin(self.underlying_price_series)]
            lowest_price = self.underlying_price_series.min()
            self.date_base_price_tuple2 = (lowest_date, lowest_price * (1.05 if self.direction == 'short' else 0.95))

    # todo: simplify try/except structure, make coherent with base_price_series property and base_price_change_per_annum
    def enforce_base_price_increase_per_annum(self, abs_increase_pa: float = .02) -> None:
        """ Enforce defined base price increase rate by keeping the lower (higher for shorts) base_price anchor point and changing the other. """
        # sign based on product's direction
        price_change_pa = np.abs(abs_increase_pa) * (-1 if self.direction == 'short' else 1)

        # if short product and the first base price inference point is higher or long product and second inference point is higher,
        # keep the first inference point:
        if (self.direction == 'short' and self.date_base_price_tuple[1] > self.date_base_price_tuple2[1]) or (
                self.direction == 'long' and self.date_base_price_tuple[1] < self.date_base_price_tuple2[1]):

            # in this case, the first tuple is the one to be kept:
            try:  # set second date one year ahead:
                date2 = self.date_base_price_tuple[0] + pd.Timedelta('364d')  # 364 equals exactly 52 weeks
                _ = self.underlying_price_series[date2]
                is_one_month = False
            except KeyError:  # if +1 Year is beyond provided data, set date back one year:
                try:
                    date2 = self.date_base_price_tuple[0] - pd.Timedelta('364d')
                    _ = self.underlying_price_series[date2]
                    is_one_month = False
                except KeyError:
                    date2 = self.date_base_price_tuple[0] - pd.Timedelta('28d')
                    _ = self.underlying_price_series[date2]
                    is_one_month = True  # smaller timedelta (half year) leads to smaller base price change in below formula:

            self.date_base_price_tuple2 = (date2, self.date_base_price_tuple[1] * ((1 + price_change_pa) ** (1/12 if is_one_month else 1)))
        else:
            # in this case, keep the second:
            try:  # set second date one year ahead:
                date = self.date_base_price_tuple2[0] + pd.Timedelta('364d')  # 364 equals exactly 52 weeks
                _ = self.underlying_price_series[date]  # try accessing date
                is_one_month = False
            except KeyError:  # if +1 Year is beyond provided data, set date back one year:
                try:
                    date = self.date_base_price_tuple2[0] - pd.Timedelta('364d')
                    _ = self.underlying_price_series[date]
                    is_one_month = False
                except KeyError:
                    date = self.date_base_price_tuple2[0] - pd.Timedelta('28d')
                    _ = self.underlying_price_series[date]
                    is_one_month = True  # smaller timedelta (half year) leads to smaller base price change in below formula:
            self.date_base_price_tuple = (date, self.date_base_price_tuple2[1] * ((1 + price_change_pa) ** (1/12 if is_one_month else 1)))

    ### String representation ###
    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return self.describe()

    def describe(self) -> str:
        intro_str = "------------------- KOCertificate Instance -------------------\n\n"
        data_str = f"Price Data Attributes:\n- start date: {self.date_index.min().strftime('%Y-%m-%d')}{' (equals issue date of product)' if self.issue_date is not None else ''}\n- end date: {self.date_index.max().strftime('%Y-%m-%d')}\n\n"
        product_str = f"Product Attributes:\n{f'- ISIN: {self.isin}\n- last base price: {self.base_price_series.iloc[-1]}\n' if self.isin is not None else ''}- type: {self.direction}\n- last leverage: {self.leverage_series.iloc[-1]}\n- risk premium (absolute): {self.risk_premium}\n- subscription ratio: {self.subscription_ratio}\n- current price: {self.price_series.iloc[-1]}\n- reached KO: {self.is_ko_series.iloc[-1]}\n\n"
        return intro_str + data_str + product_str

    ### Base price properties ###
    @property
    def date_base_price_tuple(self):
        """
        The primary base price calibration point.

        Returns
        -------
        tuple
            A tuple containing a date and corresponding base price.
        """
        return self._date_base_price_tuple

    @date_base_price_tuple.setter
    def date_base_price_tuple(self, value):
        """ Setting attribute triggers re-computation of base_price_series """
        if isinstance(value[0], str):
            date_str = value[0]
            if len(date_str) <= 10:
                date_str += " 12:00:00"  # add time suffix if necessary
                value = (date_str, value[1])
            value = (datetime.fromisoformat(date_str), value[1])   # convert to timestamp
        self._date_base_price_tuple = value
        self._base_price_series = None
        if self._date_base_price_tuple2 is not None:  # then p.a. recalculation is possible:
            self._base_price_change_per_annum = None

    @property
    def date_base_price_tuple2(self):
        """
        The secondary base price calibration point.

        Returns
        -------
        tuple
            A tuple containing a date and corresponding base price.
        """
        return self._date_base_price_tuple2

    @date_base_price_tuple2.setter
    def date_base_price_tuple2(self, value):
        """ Setting attribute triggers re-computation of base_price_series """
        if isinstance(value[0], str):
            date_str = value[0]
            if len(date_str) <= 10:
                date_str += " 12:00:00"  # add time suffix if necessary
                value = (date_str, value[1])
            value = (datetime.fromisoformat(date_str), value[1])   # convert to timestamp
        self._date_base_price_tuple2 = value
        self._base_price_series = None
        if self._date_base_price_tuple is not None:  # then p.a. recalculation is possible:
            self._base_price_change_per_annum = None

    @property
    def base_price_change_per_annum(self) -> float:
        """ Annual base price increase resulting from base price series. Typical values are between 0.02 and 0.03."""
        if self._base_price_change_per_annum is None:  # is set to none if base price series is overwritten
            start = self.base_price_series[self.date_base_price_tuple[0]]
            try:  # look one year ahead:
                end = self.base_price_series[
                    self.date_base_price_tuple[0] + pd.Timedelta('364d')]  # 364 equals exactly 52 weeks
                is_one_month = False
            except KeyError:  # if +1 Year is beyond provided data, look back one year:
                try:
                    end = self.base_price_series[self.date_base_price_tuple[0] - pd.Timedelta('364d')]
                    is_one_month = False
                except KeyError:  # if 1 year is too large look 1 month
                    end = self.base_price_series[self.date_base_price_tuple[0] - pd.Timedelta('28d')]
                    is_one_month = True
            self._base_price_change_per_annum = (end / start).item() ** (12 if is_one_month else 1) - 1  # if difference is from one month, convert to annual rate
        else:  # correct direction if attribute has already been provided
            self._base_price_change_per_annum = (np.abs(self._base_price_change_per_annum) * (-1 if self.direction == 'short' else 1)).item()
        return self._base_price_change_per_annum

    @property
    def base_price_series(self):
        """
        Time series of the interpolated base price used for pricing the future product.

        Returns
        -------
        pd.Series
            Interpolated base price series indexed by date.
        """
        if self._base_price_series is None:  # linear interpolation for base price calculation
            # amend date of base_price_tuple if too large for provided data:
            if pd.Timestamp(self.date_base_price_tuple[0]) > self.date_index.max():
                self.date_base_price_tuple = (self.date_index.max(), self.date_base_price_tuple[1])

            # get integer positions in date_index:
            if self._date_base_price_tuple2 is None:  # construct second price tuple from slope if not provided:
                try:  # set second date one year ahead:
                    date2 = self.date_base_price_tuple[0] + pd.Timedelta('364d')  # 364 equals exactly 52 weeks
                    _ = self.underlying_price_series[date2]
                    is_one_month = False
                except KeyError:  # if +1 Year is beyond provided data, set date back one year:
                    try:
                        date2 = self.date_base_price_tuple[0] - pd.Timedelta('364d')
                        _ = self.underlying_price_series[date2]
                        is_one_month = False
                    except KeyError:
                        date2 = self.date_base_price_tuple[0] - pd.Timedelta('28d')
                        _ = self.underlying_price_series[date2]
                        is_one_month = True  # smaller timedelta (4 weeks) leads to smaller base price change in below formula:

                self._date_base_price_tuple2 = (date2, self.date_base_price_tuple[1] * ((1 + self.base_price_change_per_annum) ** (1/12 if is_one_month else 1)))

            provided_dates = [self.date_index.get_loc(date) for date in
                              [self.date_base_price_tuple[0], self.date_base_price_tuple2[0]]]
            provided_prices = [self.date_base_price_tuple[1], self.date_base_price_tuple2[1]]

            # do interpolation and safe series:
            self._base_price_series = pd.Series(index=self.date_index,
                                                # we use scipy interpolate because this extrapolates linear trends beyond provided data points:
                                                data=interp1d(provided_dates, provided_prices, kind='linear',
                                                              fill_value='extrapolate')(
                                                    # interp1d object has to be called with integer indices
                                                    range(0, len(self.date_index))),
                                                name='base_price_series'
                                                )

            # prevent base prices below zero:
            self._base_price_series[self._base_price_series < 0] = 0

        return self._base_price_series

    ### General properties ###
    @property
    def issue_date(self):
        return self._issue_date
    @issue_date.setter
    def issue_date(self, value):
        """ Setting issue date re-computes base price series. """
        self._issue_date = value
        self._base_price_series = None

    @property
    def timestamp_last_ko_breach(self):
        """ Get timestamp of final KO breach for eventual base price adjustment. """
        # recompute KO mask (because is_ko_series stays at True if once KO was breached)
        if self.direction == 'long': ko_mask = (self.base_price_series >= self.underlying_price_series)
        else: ko_mask = (self.base_price_series <= self.underlying_price_series)

        # check for first co breach counted from behind:
        reverse_iloc = np.argmax(ko_mask.iloc[::-1])
        if reverse_iloc == 0:  # returns 0 either,
            # if no breach is found:
            if not ko_mask.iloc[-1]:  # which is only the case if the last element is not True
                return None
            # or if breach lasts until end:
            else:
                reverse_iloc = np.argmin(ko_mask.iloc[::-1])

        # then also reverse the int index because it's based on the reversed series:
        iloc_last_breach = len(ko_mask) - 1 - reverse_iloc

        # return timestamp:
        return np.array(self.date_index)[iloc_last_breach]

    @property
    def is_ko_series(self):
        """ True if product breached KO price. """
        # check for KO breach:
        if self.direction == 'long': ko_mask = (self.base_price_series >= self.underlying_price_series)
        else: ko_mask = (self.base_price_series <= self.underlying_price_series)

        ko_index = np.argmax(self.base_price_series >= self.underlying_price_series)
        if ko_index != 0:  # 0 is also returned if no True is found in the mask
            ko_mask.iloc[ko_index:] = True  # KO is final

        return pd.Series(ko_mask, name='Knockout Bool')

    @property
    def direction(self) -> Literal['long', 'short']:
        """
        Indicates whether the product is currently 'long' or 'short' based on final values.

        Returns
        -------
        str
            'long' if underlying > base price, 'short' if underlying < base price.
        """
        if self._direction is None:
            try:
                if self.underlying_price_series.iloc[0] > self.base_price_series.iloc[0]: self._direction = 'long'
                else: self._direction = 'short'
            except TypeError:
                raise AttributeError("For direction inference base_price_series need to be given. Provide such, run inference from historic prices or leverages methods, or manually enter the direction.")
        return self._direction
    @direction.setter
    def direction(self, value: Literal['long', 'short']):
        self._direction = value

    @property
    def leverage_series(self) -> pd.Series:
        """
        Time series representing the leverage ratio of the product.

        Returns
        -------
        pd.Series
            Leverage values computed daily.
        """
        return pd.Series(index=self.date_index,
                         # numpy to accelerate computation:
                         data=(np.array(self.underlying_price_series) * self.subscription_ratio / np.array(
                             self.price_series)
                               * np.where(self.is_ko_series, 0, 1)  # multiply with 0 if KO is breached
                               ),
                         name='Leverage')

    @property
    def date_index(self):
        """
        Returns the index of the underlying time series.

        Returns
        -------
        pd.DatetimeIndex
            Dates associated with the underlying prices.
        """
        return self.underlying_price_series.index

    ### Price calculation ###
    @property
    def underlying_price_series(self):
        """ Underlying price series, considers issue_date attribute."""
        if self.issue_date is not None:
            if self._underlying_price_series.index.min() < pd.Timestamp(self.issue_date):
                return self._underlying_price_series[self.issue_date:]  # return prices starting at issue date
        else: return self._underlying_price_series  # return all provided prices

    @property
    def intrinsic_value_series(self):
        """
        Time series of the product's intrinsic value, which is
        (underlying - base price) * subscription ratio. Includes KO check.

        Returns
        -------
        pd.Series
            Intrinsic value per date.
        """
        intrinsic_value_array = np.abs(np.array(self.underlying_price_series) - np.array(
                             self.base_price_series)) * self.subscription_ratio

        # check for first knockout event:
        ko_index = np.argmax(self.is_ko_series)
        if ko_index != 0:
            intrinsic_value_array[ko_index:] = 10 ** -10  # after knockout initial value is zero (with small increment to prevent ZeroDivisionError
        else:  # returns 0 if no KO event is found
            # or if KO event is actually at index pos. 0:
            if self.is_ko_series.iloc[0]:
                print("[WARNING] With current provided base prices, KO is breached at first date of provided underlying price series. Either adjust issue_date (to postpone starting date) or base_price properties!")
                # find last KO breach by reversing mask and looking for first True:
                print(f"\tCurrent last KO breach is at {self.timestamp_last_ko_breach} with underlying price {self.underlying_price_series[self.timestamp_last_ko_breach]}!")
                intrinsic_value_array[ko_index:] = 10 ** -10

        return pd.Series(index=self.date_index,
                         # numpy to accelerate computation:
                         data=intrinsic_value_array,
                         name='Intrinsic Value')

    @property
    def price_series(self):
        """
        Full price series of the future product, incorporating
        intrinsic value and constant risk premium.

        Returns
        -------
        pd.Series
            Final product prices over time.
        """
        price_array = np.array(self.intrinsic_value_series) + self.risk_premium

        # return as Series:
        return pd.Series(index=self.date_index,
                  data=price_array,
                  name='Future Price')


# auxiliary functions:
def fetch_future_info_from_boerse_fra(isin: str, driver_executable_path: str):
    """ Scrape risk premium, subscription ratio, base price and issue date for future product based on ISIN. """
    # auxiliary functions:
    def highlight_element(element):
        driver.execute_script(
            "arguments[0].style.border='3px solid red'; arguments[0].style.background='yellow';",
            element
        )

    def fetch_table_element(elements: [], keyword: str) -> str:
        for element in elements:
            label = element.find_element(By.CSS_SELECTOR, '.widget-table-cell').text
            if keyword not in label: continue  # search until keyword match
            element =  element.find_element(By.CSS_SELECTOR, '.widget-table-cell.text-end')
            highlight_element(element)
            return element.text
        raise KeyError(f"No element with label {keyword} found!")

    # create url (based on known structure):
    url = f"https://www.boerse-frankfurt.de/zertifikat/{isin.lower()}"

    # initialise driver and open url:
    service = Service(executable_path=driver_executable_path)
    driver = webdriver.Chrome(service=service)  # opens window, do not close!
    driver.get(url)

    # wait until page is loaded (based on presence of "button" table):
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,
                                        ".ng-star-inserted .d-flex.flex-nowrap .widget-container-v2 .content-wrapper .ng-star-inserted"))
    )

    # navigate to product information ("Stammdaten") tab:
    stammdaten_button = driver.find_elements(By.CSS_SELECTOR,
                                             ".ng-star-inserted .d-flex.flex-nowrap .widget-container-v2 .content-wrapper .ng-star-inserted")[
        8]  # .find_elements(By.CSS_SELECTOR, ".")
    highlight_element(stammdaten_button)
    stammdaten_button.click()

    # wait until new page is loaded (based on presence of "kennzahlen" table):
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,
                                        ".ng-star-inserted .d-flex .widget-container-v2 .content-wrapper .ng-star-inserted .col-12 .ar-mt .row .col-12 .table-responsive .table.widget-table .widget-table-row.ng-star-inserted"))
    )

    # locate, highlight and fetch risk_premium:
    kennzahlen_table_entries = driver.find_elements(By.CSS_SELECTOR,
                                                    '.ng-star-inserted .d-flex .widget-container-v2 .content-wrapper .ng-star-inserted .col-12 .ar-mt .row .col-12 .table-responsive .table.widget-table .widget-table-row.ng-star-inserted')
    risk_premium = fetch_table_element(kennzahlen_table_entries, "Aufgeld absolut")

    # subscription_ratio:
    basiswert_table_entries = driver.find_elements(By.CSS_SELECTOR,
                                                   '.ng-star-inserted .d-flex .widget-container-v2 .content-wrapper .ng-star-inserted .col-12 .ar-mt .row .col-12 .table-responsive .table.widget-table')[
        1].find_elements(By.CSS_SELECTOR, '.widget-table-row.ng-star-inserted')
    subscription_ratio = fetch_table_element(basiswert_table_entries, 'Bezugsverhältnis')

    # base_price and issue_date:
    stammdaten_table_entries = driver.find_elements(By.CSS_SELECTOR,
                                                    '.ng-star-inserted .d-flex .widget-container-v2 .content-wrapper .ng-star-inserted .col-12.col-lg-6.ar-half-pr-lg .widget.ar-p .row .col-12 .table-responsive .table.widget-table .widget-table-row')  # .ar-mt') .row')# .col-12')# .table-responsive .table.widget-table')[1].find_elements(By.CSS_SELECTOR, '.widget-table-row.ng-star-inserted')
    base_price = fetch_table_element(stammdaten_table_entries, 'Basispreis')
    issue_date = fetch_table_element(stammdaten_table_entries, 'Ausgabedatum')

    # close driver and return:
    driver.quit()
    return strconv.str_to_float(risk_premium), strconv.str_to_float(subscription_ratio), strconv.str_to_float(base_price), issue_date


class KOCertificateSet:
    """
    A container class for managing and visualizing a collection of `KOCertificate` instances.

    This class enables users to group multiple `KOCertificate` objects to design a comprehensive
    set of leveraged products (e.g., certificates) for analysis or comparison. If no
    `KOCertificate` instances are provided directly, the set can be initialized by specifying
    leverage parameters to automatically create products.

    Parameters
    ----------
    future_product_instances : list of KOCertificate, optional
        A list of pre-defined `KOCertificate` objects to include in the set. If not provided,
        the products will be automatically created using the specified leverage parameters.
    underlying_price_series : pd.Series, optional
        The time series of the underlying asset's price. Required if automatic product
        generation is used.
    base_price_inference_timestamp : str, optional
        List of timestamps at which to infer the base price from the desired leverage.
        Required for automatic product generation.
    n_products_per_direction : int, optional
        Number of products to generate per direction (long/short). Required if
        `future_product_instances` is not provided.
    lowest_leverage : float, optional
        The lowest leverage value to include when generating products.
    highest_leverage : float, optional
        The highest leverage value to include when generating products.

    Raises
    ------
    AttributeError
        If `future_product_instances` is not provided and required parameters for automatic
        initialization are missing.
    """

    def __init__(self,
                 future_product_instances: [KOCertificate] = None,
                 underlying_price_series: pd.Series = None,
                 base_price_inference_timestamps: [str] = None,
                 n_products_per_direction: int = None,
                 lowest_leverage: float = None,
                 highest_leverage: float = None,
                 abs_base_price_change_threshold: float = .04,
                 ):
        # private attributes:
        self._last_isin_counter = 0
        self._long_leverage_frame = self._short_leverage_frame = self._price_frame = None  # accessible through property
        self._isin_product_dict = None  # accessible through property

        if future_product_instances is not None:  # initialise products from provided list
            # convert product instances to list if necessary:
            if not isinstance(future_product_instances, list): future_product_instances = [future_product_instances]
            self._future_product_instances = future_product_instances
        else:  # or from desired leverages:
            # sanity check:
            if underlying_price_series is None or n_products_per_direction is None or lowest_leverage is None or highest_leverage is None or base_price_inference_timestamps is None:
                raise AttributeError(
                    "If no future_product_instances are provided all other arguments need to be defined for automatic product initialisation from desired leverages.")
            self._future_product_instances = self.initialise_products_from_leverage(
                underlying_price_series=underlying_price_series,
                base_price_inference_timestamps=base_price_inference_timestamps if isinstance(
                    base_price_inference_timestamps,
                    list) else [
                    base_price_inference_timestamps],
                n_products_per_direction=n_products_per_direction,
                lowest_leverage=lowest_leverage,
                highest_leverage=highest_leverage,
                abs_base_price_change_threshold=abs_base_price_change_threshold)
        # future product instances accessible through property to leverage setter for resetting _date_leverage_frame and _isin_product_dict

    def plot_leverages(self, plot_size=(15, 10), leverage_lim=(0, 10), show_legend=False):
        """
        Plot the leverage development over time for all products in the set.

        Leverage time series are displayed separately for long and short products.

        Parameters
        ----------
        plot_size : tuple of int, default (15, 10)
            The size of the plot in inches (width, height).
        leverage_lim : tuple of float, default (0, 10)
            The y-axis limits for the leverage plot.

        Returns
        -------
        None
        """
        fig, (long_ax, short_ax) = plt.subplots(2, 1, figsize=plot_size)

        # split by direction:
        long_list = [product for product in self.future_product_instances if product.direction == 'long']
        short_list = [product for product in self.future_product_instances if product.direction == 'short']

        for ax, product_list in zip([long_ax, short_ax], [long_list, short_list]):
            # prepare list of colors:
            cmap = plt.get_cmap('hsv')
            colors = cmap(np.linspace(0, 1, len(product_list)))

            for product, color in zip(product_list, colors):
                # plot each leverage:
                ax.plot(product.date_index, product.leverage_series, color=color,
                        label=f"{product.isin if product.isin is not None else 'ISIN not provided'}")

            # formatting
            if show_legend: ax.legend(loc='upper left')
            ax.set_ylabel('Leverage [x]')
            ax.set_xlabel('Date')
            ax.set_ylim(leverage_lim)
            ax.grid(color='grey')
        long_ax.set_title('Long Products')
        short_ax.set_title('Short Products')
        fig.tight_layout()

    def generate_new_isin(self, direction: Literal["long", "short"] = None) -> str:
        """ Generate new artificial ISIN while preventing duplicates. """
        if self._last_isin_counter > 9999: self._last_isin_counter = 0  # reset counter upon max displayable int
        new_isin = f"ARTIF{('LO' if direction == 'long' else 'SH') if direction is not None else '00'}{str(self._last_isin_counter).zfill(4)}"
        self._last_isin_counter += 1
        return new_isin

    def initialise_products_from_leverage(self, underlying_price_series: pd.Series,
                                          base_price_inference_timestamps: [str],
                                          n_products_per_direction: int = 5, lowest_leverage: float = 1.0,
                                          highest_leverage: float = 5.0,
                                          fix_initial_knockouts=True, abs_base_price_change_threshold: float = .04,
                                          issue_date: str = None) -> [KOCertificate]:
        """
        Generate a list of `KOCertificate` instances based on specified leverage settings.

        This method creates both long and short products with evenly spaced leverage values
        between `lowest_leverage` and `highest_leverage`. Each product's base price is
        inferred from its desired leverage at the specified timestamp.

        Parameters
        ----------
        underlying_price_series : pd.Series
            The time series of the underlying asset's price.
        base_price_inference_timestamps : [str]
            List of timestamps at which to infer the base price from the desired leverage.
        n_products_per_direction : int, default 5
            The number of products to create for each direction (long and short).
        lowest_leverage : float, default 1.0
            The lowest leverage value for generated products.
        highest_leverage : float, default 5.0
            The highest leverage value for generated products.

        Returns
        -------
        list of KOCertificate
            A list of initialized `KOCertificate` instances with desired leverages.
        """
        # direction and leverage lists:
        desired_directions = ['long'] * n_products_per_direction + ['short'] * n_products_per_direction
        desired_leverages = list(np.linspace(lowest_leverage, highest_leverage, n_products_per_direction)) * 2

        # initialise product list:
        product_list = []
        # iterate over timestamps at which to guarantee leverages, i.e. initialise base prices:
        for base_price_inference_timestamp in base_price_inference_timestamps:
            # initialise products
            temp_product_list = [KOCertificate(underlying_price_series=underlying_price_series,
                                               isin=self.generate_new_isin(direction=dir),
                                               scrape_data_if_possible=False, direction=dir, issue_date=issue_date) for
                                 dir in desired_directions]
            # infer base price from leverages at base_price:
            for ind, (product, leverage) in enumerate(tqdm(zip(temp_product_list, desired_leverages))):
                product.get_base_price_from_leverage(date=base_price_inference_timestamp, leverage=leverage,
                                                     use_as_1st_base_price_tuple=True)

                # parametrise base price development:
                if fix_initial_knockouts:
                    product.fix_initial_knockout()  # resolves initial KOs, does nothing if no initial KO
                if abs_base_price_change_threshold is not None:  # if (often from resolving initial KOs) the base price change is too steep:
                    if np.abs(product.base_price_change_per_annum) > abs_base_price_change_threshold:
                        product.enforce_base_price_increase_per_annum(abs_base_price_change_threshold)  # adjust such

            product_list += temp_product_list  # append to product list

        return product_list

    def get_product_from_leverage_span(self, date: str, direction: Literal['long', 'short'],
                                       leverage_span: (int, int),
                                       return_all=False,
                                       search_ascending=True) -> Union[str, KOCertificate]:

        """ Search for product with the smallest (if search_ascending) leverage inside leverage_span. Raises KeyError if no produt found. """
        # utilise pre-computed leverage frame across time and products:
        frame = self.long_leverage_frame if direction == 'long' else self.short_leverage_frame
        leverages = frame.loc[date, :]  # select specified date
        candidates = leverages.loc[
            (leverages >= leverage_span[0]) & (leverages <= leverage_span[1])]  # leverage span condition
        if len(candidates) == 0: raise KeyError("No product with respective leverage found!")

        # select product with the smallest leverage inside span:
        if not return_all:
            isin = candidates.sort_values(ascending=search_ascending).index[0]
            return isin
        else:  # if all isins should be returned:
            isins = list(candidates.sort_values(ascending=search_ascending).index)
            if not isinstance(isins, list): isins = [isins]
            return isins

    @staticmethod
    def _check_leverage_availability_per_row(row: pd.Series, leverage_categories: [float] = None,
                                             include_open_leverage_category: bool = False,
                                             # if True, last category is open to inf.
                                             ) -> pd.Series:
        """ To be applied via pd.DataFrame.apply(axis='columns') to each row of the leverage_frame. """
        # construct leverage categories to check:
        if leverage_categories is None: leverage_categories = [1.0, 2.0, 3.0, 4.0, 5.0]
        leverage_span_tuples = [(start, np.inf if ind + 1 == len(leverage_categories) else leverage_categories[ind + 1])
                                for ind, start in enumerate(leverage_categories)]

        # remove last category if required:
        if not include_open_leverage_category:
            leverage_span_tuples = leverage_span_tuples[:-1]
            leverage_categories = leverage_categories[:-1]

        # derive unique leverages:
        unique_leverages = pd.Series(row.unique())

        # return series with Bool whether leverage is available per category:
        availabilities = [len(unique_leverages.loc[(upper > unique_leverages) & (unique_leverages > lower)]) != 0 for
                          lower, upper in leverage_span_tuples]
        return pd.Series(index=leverage_categories, data=availabilities)

    def get_leverage_availability(self, product_type: Literal["long", "short"],
                                  hour_minute_to_check: (int, int) = None,
                                  leverage_categories: [float] = None,
                                  include_open_leverage_category: bool = False,
                                  # if True, last category is open to inf.
                                  verbose=True) -> pd.DataFrame:
        """ Check leverage product availability along each time step (rows) and leverage_category (columns). """
        # select leverage_frame based on direction:
        if product_type == "long":
            frame = self.long_leverage_frame
        elif product_type == "short":
            frame = self.short_leverage_frame

        # select timestamps to be scrutinized:
        frame = frame.loc[
            (frame.index.hour == hour_minute_to_check[0]) & (frame.index.minute == hour_minute_to_check[1])]

        # check leverage availabilities:
        avail_frame = frame.apply(func=self._check_leverage_availability_per_row, axis='columns',
                                  # function args:
                                  leverage_categories=leverage_categories,
                                  include_open_leverage_category=include_open_leverage_category)

        # print statement:
        if verbose:
            for column in avail_frame:
                print(f'Availability of leverage >{column}:',
                      avail_frame[column].value_counts()[True] / len(avail_frame) * 100, "%")

        return avail_frame

    ######### Properties #######
    @property
    def future_product_instances(self) -> [KOCertificate]:
        """ List of included future products. Changing resets private attributes _date_leverage_frame and _isin_product_dict. """
        return self._future_product_instances

    @future_product_instances.setter
    def future_product_instances(self, value: [KOCertificate]):
        self._future_product_instances = value
        self._isin_product_dict = None
        self._long_leverage_frame = self._short_leverage_frame = self._price_frame = None

    @property
    def by_isin(self) -> {str: KOCertificate}:
        """ Locate product from set by ISIN. """
        if self._isin_product_dict is None:
            self._isin_product_dict = {product.isin: product for product in self.future_product_instances}
        return self._isin_product_dict

    @property
    def price_frame(self) -> pd.DataFrame:
        """ Dataframe with all product's prices by date (rows) and product-isin (columns). """
        if self._price_frame is None:
            self._price_frame = pd.DataFrame(
                {product.isin: product.price_series for product in self.future_product_instances})
        return self._price_frame

    @property
    def long_leverage_frame(self) -> pd.DataFrame:
        """ Dataframe with long products' leverages by date (rows) and product-isin (columns). """
        if self._long_leverage_frame is None:
            self._long_leverage_frame = pd.DataFrame(
                {product.isin: product.leverage_series for product in self.future_product_instances if
                 product.direction == 'long'})
            self._long_leverage_frame.fillna(0,
                                             inplace=True)  # na arises if issue date is later than considered timestamp, 0 is used as identifier that product isn't available (either KO or not yet issued)
        return self._long_leverage_frame

    @property
    def short_leverage_frame(self) -> pd.DataFrame:
        """ Dataframe with short products' leverages by date (rows) and product-isin (columns). """
        if self._short_leverage_frame is None:
            self._short_leverage_frame = pd.DataFrame(
                {product.isin: product.leverage_series for product in self.future_product_instances if
                 product.direction == 'short'})
            self._short_leverage_frame.fillna(0,
                                              inplace=True)  # na arises if issue date is later than considered timestamp, 0 is used as identifier that product isn't available (either KO or not yet issued)
        return self._short_leverage_frame

    @property
    def leverage_frame(self) -> pd.DataFrame:
        """ Dataframe with all products' leverages by date (rows) and product-isin (columns). """
        return pd.concat([self.long_leverage_frame, self.short_leverage_frame], axis='columns')