import src.utils.str_conversion as strconv
import src.utils.file_management as filemgmt

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