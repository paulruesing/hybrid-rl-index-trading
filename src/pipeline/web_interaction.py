import src.utils.str_conversion as strconv
import src.utils.file_management as filemgmt

import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pathlib import Path
import pandas as pd

def fetch_future_info_from_boerse_fra(isin: str, driver_executable_path: str):
    """
    Fetches and extracts details about a financial product from the Börse Frankfurt website using
    a web scraping approach.

    This function retrieves specific data points for the given financial product:
    risk premium, subscription ratio, base price, and issue date. The function uses a web driver
    to navigate through pages, interact with elements, and extract the required data.

    The function leverages auxiliary functions, element highlighting, and CSS selectors
    to perform targeted data scraping, ensuring accurate data fetching from the website's structure.

    Parameters
    ----------
    isin : str
        The International Securities Identification Number (ISIN) of the financial product
        to fetch data for.
    driver_executable_path : str
        The file path for the web driver executable required to control the browser.

    Returns
    -------
    tuple[float, float, float, str]
        A tuple containing the following:
        - risk_premium (float): The absolute risk premium of the financial product.
        - subscription_ratio (float): The subscription ratio of the financial product.
        - base_price (float): The base price of the financial product.
        - issue_date (str): The issue date of the financial product in string format.
    """
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
                                             ".ng-star-inserted .d-flex.flex-nowrap .widget-container-v2 .content-wrapper .ng-star-inserted")[7]  # .find_elements(By.CSS_SELECTOR, ".")
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


def fetch_price_from_comdirect(raw_download_dir: Path,
                               url: str = "https://www.comdirect.de/inf/fonds/detail/chart.html?ID_NOTATION=115802659&",
                               driver_executable_path: str = "", verbose: bool = True) -> pd.Series:
    """
    Fetches daily price data from Comdirect and returns it as a pandas Series.

    This function uses Selenium WebDriver to navigate the Comdirect website, accept cookies,
    and download historical price data in CSV format. The downloaded data is then processed
    and returned as a pandas Series with datetime index.

    Parameters
    ----------
    raw_download_dir : Path
        Directory where the downloaded CSV file will be stored.
    url : str, optional
        URL of the Comdirect page to fetch the data from. Default is a specific fund page for DAX ETF.
        Beware, functioning with different URLs hasn't been tested yet.
    driver_executable_path : str, optional
        Path to the Selenium WebDriver executable for Chrome.
    verbose : bool, optional
        If True, prints a message upon successful completion.

    Returns
    -------
    pd.Series
        A pandas Series with datetime index representing the price data sorted in ascending order.

    Raises
    ------
    TimeoutException
        If any of the web elements fail to load within the specified wait time using Selenium.
    FileNotFoundError
        If the downloaded CSV file is not found in the specified directory.
    ValueError
        If the downloaded CSV file does not contain valid data for processing.
    """
    # download directory
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": str(raw_download_dir),
             "download.prompt_for_download": False,
             "download.directory_upgrade": True,
             "safebrowsing.enabled": True}
    chrome_options.add_experimental_option("prefs", prefs)

    service = Service(executable_path=driver_executable_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)  # opens window, do not close!
    driver.get(url)

    # wait until page is loaded (based on presence of cookie pop-up):
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/com-consent-layer"))
    )

    # accept cookie button (is hidden in shadow-subtree)
    pop_up = driver.find_elements(By.XPATH, "/html/body/com-consent-layer")[0]
    pop_up = pop_up.shadow_root  # access shadow subtree

    # locate and click
    cookie_button = pop_up.find_element(By.CSS_SELECTOR,
                                        "dialog div div com-button-area com-button")  # > div:nth-child(2) > div:nth-child(2) > com-button-area > com-button:nth-child(1)")
    cookie_button.click()

    # open quote list button:
    quote_button = driver.find_elements(By.ID, "openQuoteListButton")[0]
    quote_button.click()

    # wait until download pop-up is loaded (based on presence of "button" table):
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[15]/div/div[2]/div/div/div/div/div/div/div[2]/div/a"))
    )

    # download quotes:
    download_button = \
    driver.find_elements(By.XPATH, "/html/body/div[15]/div/div[2]/div/div/div/div/div/div/div[2]/div/a")[
        0]
    download_button.click()

    if verbose: print("Successfully downloaded price data from comdirect.")

    # fetch download:
    downloaded_frame = pd.read_csv(filemgmt.most_recent_file(raw_download_dir, ".csv", search_by='meta-data'),
                                   sep=";", encoding="latin-1")
    # format frame:
    formatted_frame = downloaded_frame.reset_index().iloc[1:, :]
    formatted_frame.rename(columns={'level_0': 'date', 'level_1': 'time', 'level_3': 'price'}, inplace=True)

    # type conversion:
    formatted_frame['datetime'] = pd.to_datetime(
        formatted_frame['date'].astype(str) + ' ' + formatted_frame['time'].astype(str), dayfirst=True)
    formatted_frame['price'] = pd.to_numeric(formatted_frame['price'].str.replace(',', '.'), errors='coerce')

    # return sorted series:
    price_series = formatted_frame.set_index('datetime')['price']
    return price_series.sort_index()


def wait_for_then_locate_element(driver: webdriver.Chrome, xpath: str, timeout: int = 20):
    """
    Waits for a web element to be present in the DOM, located by the given XPath, before returning the element.

    Parameters
    ----------
    driver : webdriver.Chrome
        The web driver instance controlling the browser session.
    xpath : str
        The XPath of the web element to locate.
    timeout : int, optional
        The maximum number of seconds to wait for the element to be located (default is 20).

    Returns
    -------
    WebElement
        The located web element once it becomes available in the DOM.

    Raises
    ------
    TimeoutException
        If the web element is not located within the specified timeout.
    """
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    return driver.find_element(By.XPATH, xpath)


def login_to_wikifolio(username: str, password: str, wikifolio_id: str = "wfprtresen",
                       driver_executable_path="") -> webdriver.Chrome:
    """
    Logs into the specified Wikifolio account and navigates to the desired page.

    This function automates the process of logging into a Wikifolio account using a given set
    of credentials and navigates to a specific Wikifolio ID page using a Selenium-based web driver.
    The function also handles common popups like cookie notices and region confirmation dialogs.

    Parameters
    ----------
    username : str
        The username or email address used for logging into the Wikifolio account.
    password : str
        The password associated with the specified username.
    wikifolio_id : str, optional
        The ID of the target Wikifolio page to be accessed after login.
        Defaults to "wfprtresen".
    driver_executable_path : str, optional
        Path to the ChromeDriver executable. If not provided, Selenium will use the default
        configured executable.

    Returns
    -------
    webdriver.Chrome
        An instance of the active Chrome WebDriver session after a successful login.
    """
    url = f"https://www.wikifolio.com/de/de/meine-wikifolios/trade/{wikifolio_id.lower()}"

    service = Service(executable_path=driver_executable_path)
    driver = webdriver.Chrome(service=service)  # opens window, do not close!
    driver.get(url)

    # cookie button:
    wait_for_then_locate_element(driver, "/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[2]").click()

    # region button (appears after mouse movement)
    time.sleep(1)
    actions = webdriver.ActionChains(driver)
    actions.move_by_offset(10, 5).perform()  # moves mouse 10px right, 5px down from current position
    wait_for_then_locate_element(driver, "/html/body/div[7]/div[3]/div/section/footer/div/button").click()

    # login query button:
    wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/div[1]/button").click()

    # input login data:
    wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/form/div/div[1]/input").send_keys(
        username)
    wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/form/div/div[2]/input").send_keys(
        password)
    # submit:
    wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/form/div/button").click()

    return driver


def add_products_to_wikifolio(driver: webdriver.Chrome, isins_to_add: list[str]):
    """
    Adds a list of products identified by their ISINs to a wikifolio.

    Parameters
    ----------
    driver : webdriver.Chrome
        An instance of Chrome WebDriver. The driver must be logged in to wikifolio.
    isins_to_add : list[str]
        A list of ISINs to add to the wikifolio. If an ISIN is already included, no action will be taken for it.

    """
    ### add isins from provided list (if already included, doesn't matter)
    # driver has to be logged into wikifolio already
    if isins_to_add is not None:
        for isin in isins_to_add:
            add_isin_input = wait_for_then_locate_element(driver,
                                                          "/html/body/div[3]/main/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div/div[2]/div/span[1]/input[2]")
            add_isin_input.send_keys(isin)  # slow down to prevent errors
            add_product_button = wait_for_then_locate_element(driver,
                                                              "/html/body/div[3]/main/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div/div[2]/div/span[1]/div/div/div[2]")
            add_product_button.click()
            time.sleep(2)  # slow down to prevent errors


def scrape_portfolio_holdings_from_wikifolio(driver: webdriver.Chrome) -> tuple[float, float, dict[str, float]]:
    """
    Scrapes portfolio holdings from a logged-in Wikifolio account.

    This function extracts data about the cash balance, total portfolio value, and shares per ISIN from a specific table on the Wikifolio website. It requires the user to already be logged into their Wikifolio account before calling this function.

    Parameters
    ----------
    driver : webdriver.Chrome
        A Selenium WebDriver instance that must already be logged into Wikifolio and pointing to the relevant portfolio page.

    Returns
    -------
    tuple[float, float, dict[str, float]]
        A tuple containing:
        - `wf_cash` : The cash balance in the portfolio.
        - `wf_total` : The total value of the portfolio.
        - `shares_p_isin_dict` : A dictionary mapping ISINs (International Securities Identification Numbers) to the number of shares held for each.
    """
    # driver has to be logged into wikifolio (login_to_wikifolio method) already

    ### fetch shares per isin:
    shares_p_isin_dict = {}
    # locate relevant table:
    product_table = wait_for_then_locate_element(driver,
                                                 "/html/body/div[3]/main/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div/div/div[1]/div/div/table/tbody")
    product_table_rows = product_table.find_elements(By.XPATH, "tr")
    for row_ind, row in enumerate(product_table_rows):
        # skip irrelevant rows
        if row_ind == 0:
            continue  # column label row
        elif row_ind == len(product_table_rows) - 2:  # cash
            wf_cash_entry = row.find_element(By.XPATH, "td[4]/div")
            wf_cash = strconv.str_to_float(wf_cash_entry.text)
            continue
        elif row_ind == len(product_table_rows) - 1:  # portfolio value
            wf_total_entry = row.find_element(By.XPATH, "td[4]/div")
            wf_total = strconv.str_to_float(wf_total_entry.text)
            continue

        # fetch and convert relevant entries:
        isin_entry = row.find_element(By.XPATH, "td[1]/div/div/div")
        isin = isin_entry.text
        shares_count_entry = row.find_element(By.XPATH, "td[3]/div")
        shares_count = strconv.str_to_float(shares_count_entry.text)

        shares_p_isin_dict[isin] = shares_count

    return wf_cash, wf_total, shares_p_isin_dict