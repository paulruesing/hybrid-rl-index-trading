import src.utils.str_conversion as strconv
import src.utils.file_management as filemgmt

from typing import Union
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.select import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException

from pathlib import Path
import pandas as pd

def get_driver_options(raw_download_dir: Union[Path, str] = None) -> webdriver.ChromeOptions:
    """
    Configures and returns ChromeDriver options, allowing customization for download directory, headless execution, and other browser settings.

    Parameters
    ----------
    raw_download_dir : Union[Path, str], optional
        Path to the directory where download files should be saved. If specified, the browser will save files to this directory without prompting the user. If None, default browser behavior is applied.

    Returns
    -------
    webdriver.ChromeOptions
        Configured ChromeDriver options object.
    """
    chrome_options = webdriver.ChromeOptions()

    if raw_download_dir is not None:  # set download dir of driver instance
        prefs = {"download.default_directory": str(raw_download_dir),
                 "download.prompt_for_download": False,
                 "download.directory_upgrade": True,
                 "safebrowsing.enabled": True}
        chrome_options.add_experimental_option("prefs", prefs)

    # enable background run:
    #chrome_options.add_argument("--headless")  # Enables headless mode
    #chrome_options.add_argument("--disable-gpu")  # Disables GPU rendering (useful for older versions Windows)
    # currently leads to issues!

    chrome_options.add_argument("--window-size=1920x1080")  # Sets window size to avoid rendering issues

    return chrome_options


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
    driver = webdriver.Chrome(service=service, options=get_driver_options())  # opens window, do not close!
    driver.get(url)

    # Set zoom to 50% (zoom out) to prevent display size issues
    driver.execute_script("document.body.style.zoom='50%'")

    # wait until page is loaded (based on presence of "button" table):
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,
                                        ".ng-star-inserted .d-flex.flex-nowrap .widget-container-v2 .content-wrapper .ng-star-inserted"))
    )

    # navigate to product information ("Stammdaten") tab:
    stammdaten_button = driver.find_elements(By.CSS_SELECTOR,
                                             ".ng-star-inserted .d-flex.flex-nowrap .widget-container-v2 .content-wrapper .ng-star-inserted")[7]  # .find_elements(By.CSS_SELECTOR, ".")
    highlight_element(stammdaten_button)
    safe_click_element(driver, stammdaten_button)

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
    service = Service(executable_path=driver_executable_path)
    driver = webdriver.Chrome(service=service, options=get_driver_options(raw_download_dir))  # opens window, do not close!
    driver.get(url)

    # Set zoom to 50% (zoom out) to prevent display size issues
    driver.execute_script("document.body.style.zoom='50%'")

    # wait until page is loaded (based on presence of cookie pop-up):
    pop_up = wait_for_then_locate_element(driver, "/html/body/com-consent-layer")
    pop_up = pop_up.shadow_root  # access shadow subtree
    cookie_button = pop_up.find_element(By.CSS_SELECTOR,
                                        "dialog div div com-button-area com-button")  # > div:nth-child(2) > div:nth-child(2) > com-button-area > com-button:nth-child(1)")
    safe_click_element(driver, cookie_button)

    # open quote list button:
    quote_button = wait_for_then_locate_element(driver, id="openQuoteListButton")
    safe_click_element(driver, quote_button)

    # amend time resolution of quotes:
    interval_dropdown_element = wait_for_then_locate_element(driver, xpath="""//*[@id="FORM_KURSDATEN"]/div[2]/div/div[2]/div[2]/div/div/select""")
    interval_dropdown = Select(interval_dropdown_element)
    interval_dropdown.select_by_visible_text("15 Minuten")

    update_button = wait_for_then_locate_element(driver, """//*[@id="FORM_KURSDATEN"]/div[3]/div/button""")
    safe_click_element(driver, update_button)
    time.sleep(2)
    # wait until download pop-up is loaded (based on presence of "button" table):
    download_button = wait_for_then_locate_element(driver, xpath="""//*[@id="id_pricedata-layer_trigger-aria-description-wrapper"]/div[2]/div/div/div/div/div/div/div[2]/div/a""")
    safe_click_element(driver, download_button)
    time.sleep(10)  # wait until download is done

    if verbose: print("Successfully downloaded price data from comdirect.")

    # fetch download:
    downloaded_frame = pd.read_csv(filemgmt.most_recent_file(raw_download_dir, ".csv", search_by='meta-data'),
                                   sep=";", encoding="latin-1")

    # format frame:
    formatted_frame = downloaded_frame.reset_index().iloc[1:, :]
    # close price is level_5:
    formatted_frame.rename(columns={'level_0': 'date', 'level_1': 'time', 'level_5': 'price'}, inplace=True)

    # type conversion:
    formatted_frame['datetime'] = pd.to_datetime(
        formatted_frame['date'].astype(str) + ' ' + formatted_frame['time'].astype(str), dayfirst=True)
    formatted_frame['price'] = pd.to_numeric(formatted_frame['price'].str.replace(',', '.'), errors='coerce')

    # return sorted series:
    price_series = formatted_frame.set_index('datetime')['price']
    return price_series.sort_index()

def safe_click_element(driver: webdriver.Chrome, element):
    """
    Attempts to safely click on a web element by scrolling it into view and retrying up to three times
    before raising an exception. This is useful for handling cases where elements are not immediately
    clickable due to overlays or timing issues.

    Parameters
    ----------
    driver : webdriver.Chrome
        The instance of the Chrome WebDriver interacting with the web page.
    element
        The web element to be clicked.

    Raises
    ------
    ElementClickInterceptedException
        Raised if the element cannot be clicked successfully after three attempts.
    """
    driver.execute_script("arguments[0].scrollIntoView(true);", element)

    # 3 attempts to click:
    for attempt in range(3):
        try:
            element.click()
            return
        except ElementClickInterceptedException:
            time.sleep(2)

    # otherwise raise error:
    raise ElementClickInterceptedException("Failed to click element.")

def wait_for_then_locate_element(driver: webdriver.Chrome, xpath: str = None, id: str = None, timeout: int = 20):
    """
    Waits for an element to become present in the DOM and then locates it using either its
    XPath or ID. This function employs WebDriverWait to ensure the desired web element
    is available before attempting to locate it. If neither `xpath` nor `id` is provided, it
    raises a ValueError.

    Parameters
    ----------
    driver : webdriver.Chrome
        The Selenium WebDriver instance used to interact with the browser.
    xpath : str, optional
        The XPath of the web element to locate. Defaults to None.
    id : str, optional
        The ID of the web element to locate. Defaults to None.
    timeout : int, optional
        The maximum time to wait (in seconds) for the element to appear in the DOM.
        Defaults to 20 seconds.

    Returns
    -------
    WebElement
        The located web element using the provided XPath or ID.

    Raises
    ------
    ValueError
        If neither `xpath` nor `id` is provided.
    """
    if xpath is not None:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return driver.find_element(By.XPATH, xpath)
    elif id is not None:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, id))
        )
        return driver.find_element(By.ID, id)
    else:
        raise ValueError("Either xpath or id must be provided!")


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
    driver = webdriver.Chrome(service=service, options=get_driver_options())  # opens window, do not close!
    driver.get(url)

    # Set zoom to 50% (zoom out) to prevent display size issues
    driver.execute_script("document.body.style.zoom='50%'")

    # cookie button:
    cookie_button = wait_for_then_locate_element(driver, "/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[2]")
    safe_click_element(driver, cookie_button)

    # region button (appears after mouse movement)
    time.sleep(1)
    actions = webdriver.ActionChains(driver)
    actions.move_by_offset(10, 5).perform()  # moves mouse 10px right, 5px down from current position
    region_button = wait_for_then_locate_element(driver, "/html/body/div[7]/div[3]/div/section/footer/div/button")
    safe_click_element(driver, region_button)

    # login query button:
    login_query_button = wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/div[1]/button")
    safe_click_element(driver, login_query_button)

    # input login data:
    wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/form/div/div[1]/input").send_keys(
        username)
    wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/form/div/div[2]/input").send_keys(
        password)
    # submit:
    submit_button = wait_for_then_locate_element(driver, "/html/body/div[5]/div[3]/div/section/div/form/div/button")
    safe_click_element(driver, submit_button)

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
    # Set zoom to 50% (zoom out) to prevent display size issues
    driver.execute_script("document.body.style.zoom='50%'")

    ### add isins from provided list (if already included, doesn't matter)
    # driver has to be logged into wikifolio already
    if isins_to_add is not None:
        for isin in isins_to_add:
            add_isin_input = wait_for_then_locate_element(driver,
                                                          "/html/body/div[3]/main/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div/div[2]/div/span[1]/input[2]")
            add_isin_input.send_keys(isin)  # slow down to prevent errors
            add_product_button = wait_for_then_locate_element(driver,
                                                              "/html/body/div[3]/main/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div/div[2]/div/span[1]/div/div/div[2]")
            safe_click_element(driver, add_product_button)
            time.sleep(2)  # slow down to prevent errors


def scrape_portfolio_holdings_from_wikifolio(driver: webdriver.Chrome) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """
    Scrapes portfolio holdings data, including cash, total portfolio value, and holdings details, from a logged-in Wikifolio account.

    Parameters
    ----------
    driver : webdriver.Chrome
        Selenium WebDriver instance logged into a Wikifolio account.

    Returns
    -------
    tuple
        A tuple containing the following:
        - wf_cash (float): The cash amount available in the portfolio.
        - wf_total (float): The total value of the portfolio.
        - shares_p_isin_dict (dict[str, float]): A dictionary mapping ISINs to the count of shares held.
        - price_p_isin_dict (dict[str, float]): A dictionary mapping ISINs to the price per share.

    Notes
    -----
    This method assumes that the WebDriver has already been logged into a Wikifolio account using an appropriate login method.
    """
    # Set zoom to 50% (zoom out) to prevent display size issues
    driver.execute_script("document.body.style.zoom='50%'")

    # driver has to be logged into wikifolio (login_to_wikifolio method) already

    # prepare per isin dicts:
    shares_p_isin_dict = {}; price_p_isin_dict = {}

    # locate relevant table:
    product_table = wait_for_then_locate_element(driver,
                                                 "/html/body/div[3]/main/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div/div/div[1]/div/div/table/tbody")
    product_table_rows = product_table.find_elements(By.XPATH, "tr")

    # iterate over table rows to fetch values:
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

        price_entry = row.find_element(By.XPATH, "td[2]/span/div")
        try:
            price = strconv.str_to_float(price_entry.text)
        except ValueError:
            print(f"Error in price conversion for ISIN {isin}. Wikifolio seems to not display such.")
            price = None

        shares_count_entry = row.find_element(By.XPATH, "td[3]/div")
        shares_count = strconv.str_to_float(shares_count_entry.text)

        shares_p_isin_dict[isin] = shares_count
        price_p_isin_dict[isin] = price

    return wf_cash, wf_total, shares_p_isin_dict, price_p_isin_dict