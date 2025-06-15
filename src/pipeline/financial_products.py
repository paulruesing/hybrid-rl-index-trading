import os
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import src.utils.file_management as filemgmt
import src.pipeline.preprocessing as preprocessing
import src.utils.str_conversion as strconv


class FutureProduct:
    """ Models the price evolution of a future financial product. """
    def __init__(self,
                 underlying_price_series: pd.Series,
                 isin: str = None,
                 risk_premium: float = 0,
                 date_base_price_tuple: (str, float) = None,
                 date_base_price_tuple2: (str, float) = None,
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

    ### String representation ###
    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return self.describe()

    def describe(self) -> str:
        intro_str = "------------------- FutureProduct Instance -------------------\n\n"
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
        self._date_base_price_tuple = value
        self._base_price_series = None

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
        self._date_base_price_tuple2 = value
        self._base_price_series = None

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
            if self.date_base_price_tuple2 is not None:
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
            else:
                self._base_price_series = pd.Series(index=self.date_index, name='base_price_series')
                self._base_price_series.iloc[:] = self.date_base_price_tuple[1]

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
    def is_ko_series(self):
        """ True if product breached KO price. """
        _ = self.price_series  # KO is checked during price calculation
        return self._is_ko_series

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
            if self.underlying_price_series.iloc[0] > self.base_price_series.iloc[0]: self._direction = 'long'
            else: self._direction = 'short'
        return self._direction

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
                             self.price_series) * np.array(~self.is_ko_series)),
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

        # check for knockout event:
        if self.direction == 'long': ko_mask = (self.base_price_series >= self.underlying_price_series)
        else: ko_mask = (self.base_price_series <= self.underlying_price_series)
        ko_index = np.argmax(self.base_price_series >= self.underlying_price_series)
        if ko_index != 0:  # 0 is also returned if no True is found in the mask
            ko_mask.iloc[ko_index:] = True  # KO is final
            intrinsic_value_array[ko_index:] = 0  # price remains at risk_premium after KO breach
        self._is_ko_series = ko_mask

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