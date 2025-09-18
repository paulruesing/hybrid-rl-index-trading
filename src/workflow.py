from src.pipeline.preprocessing import StockPriceDataManager
from src.pipeline.predictors import PredictorManager, predictor_parametrisation_loop
from src.pipeline.rl_environments import RLTradingEnv, env_parametrisation_loop, TradeImplementor
from src.pipeline.rl_agents import MultiProductAgent
from src.pipeline.financial_products import KOCertificate, KOCertificateSet
from src.pipeline.chatbot import WhatsAppChatbot
from src.utils.function_decorators import retry_decorator, timed_callback_decorator
import src.pipeline.web_interaction as webinteraction
import src.utils.file_management as filemgmt

from datetime import datetime
import time
from typing import Union, Literal
from itertools import combinations
import numpy as np
import pandas as pd
import json
import os
import requests
from dotenv import load_dotenv
import logging
from typing import Union
from pathlib import Path
from ctypes import c_char
import multiprocessing
import threading

from src.utils.chatbot_app import create_app

###################### VARIABLES ######################
ROOT = Path().resolve().parent

DATA = ROOT / "data"
DOWNLOADED_PRICES = DATA / "minutely_price_downloads"
INTERPOLATED_PRICES = DATA / "interpolated_prices_dax"
SCRAPED_RAW_DOWNLOADS = DATA / "scraped_raw_downloads_archive"

SAVED_MODELS = DATA / "saved_models"
SAVED_LSTM_MODELS = SAVED_MODELS / "lstm"
SAVED_TF_MODELS = SAVED_MODELS / "transformer"
WORKING_DIR_PREDICTORS = SAVED_MODELS / "working_dir_pred_manager"

SAVED_PORTFOLIOS = DATA / "portfolios"

SAVED_BACKTESTS = DATA / "backtests"

TRAINING_LOGS = ROOT / "output" / "rl_training_logs"
PREDICTION_PLOTS = ROOT / "output" / "prediction_plots"

PRIVATE_FILES = ROOT / "private"
AV_API_KEY_FILE = PRIVATE_FILES / "Alpha Vantage API Key.txt"
with open(AV_API_KEY_FILE) as file: AV_API_KEY = file.read()

WIKIFOLIO_KEY_FILE = PRIVATE_FILES / "Wikifolio key.txt"
with open(WIKIFOLIO_KEY_FILE) as file: WIKIFOLIO_USER_KEY = file.read().split('\n')

CHATBOT_ENV_FILE = PRIVATE_FILES / "chatbot.env"
if not load_dotenv(CHATBOT_ENV_FILE): raise ValueError("Failed to load .env file")


###################### INITIALISATION ######################
data_manager = StockPriceDataManager(ticker_symbol='DAX',
                                    download_dir=DOWNLOADED_PRICES,
                                    interpolated_files_dir=INTERPOLATED_PRICES,
                                    alpha_vantage_api_key=AV_API_KEY,
                                    env_sampling_rate_minutes=15,
                                    is_etf_price_data=True,
                                    non_etf_time_price_tuples=[('2025-06-04 18:00:00', 24276.48),
                                                               ('2025-06-02 17:00:00', 23942.52),
                                                               ('2025-06-03 12:00:00', 23962.40),
                                                               ('2025-05-30 17:00:00', 23997.71)],
                                    include_scraped_downloads=False, #(scenario == "deployment"),
                                     scrape_raw_download_dir=SCRAPED_RAW_DOWNLOADS,
                                    )

pred_manager = PredictorManager(data_manager=data_manager, initialisation_dir=SAVED_MODELS, recursive=True,
                                not_older_than_n_days=7,  # only recently fine-tuned models
                                )

portfolio = KOCertificateSet.load_from_csv(
    file_path=filemgmt.most_recent_file(SAVED_PORTFOLIOS, ".csv", ["Scraped"]),
    underlying_price_series=data_manager.non_etf_env_interp_prices, )

backtest_portfolio = KOCertificateSet.load_from_csv(
        file_path=filemgmt.most_recent_file(SAVED_PORTFOLIOS, ".csv", ["Artificial"]),
        underlying_price_series=data_manager.non_etf_env_interp_prices, )

env = RLTradingEnv(price_series=data_manager.env_interp_prices,
                   price_sampling_rate_minutes=data_manager.env_sampling_rate_minutes,
                   predictor_instances=pred_manager.get_predictors_by_type_sorted(architecture='LSTM', preset_type='c2', return_instances=True,
                                                                                  k_best=1) + pred_manager.get_predictors_by_type_sorted(architecture='LSTM', preset_type='d2', return_instances=True,
                                                                                                                                         k_best=1) + pred_manager.get_predictors_by_type_sorted(architecture='LSTM', preset_type='d3', return_instances=True,
                                                                                                                                                                                                k_best=1),
                   product_set=portfolio,
                   trading_quantity_per_leverage_factor=2.5,
                   sell_opposite_direction_if_no_cash=True, sell_all_opposite_products_if_no_cash=True,
                   include_open_leverage_category=False,
                   precalculate_predictor_observations=False, # (scenario == 'backtest'),  # FALSE MAKES SENSE ONLY FOR REDUCED N_STEPS IN BACKTEST
                   leverage_categories=(1.0, 2.0, 3.0, 4.0, 5.0))  #, 5.5, 6.0, 6.5, 7.0))

agent = MultiProductAgent(observation_types=env.observation_types,
                          observation_horizons_minute=env.observation_horizons_minutes,
                          n_leverage_categories=len(env.leverage_categories),
                          abs_potential_threshold_steps=(.0025, .04, .1),  # (.04, .15, .7)
                          include_open_leverage_category=env.include_open_leverage_category,
                          potential_treshold_horizon_days=env.potential_horizon_days,
                          )

chatter = WhatsAppChatbot("../private/chatbot.env", verbose=True)


###################### DESCRIPTIVE FUNCTIONS ######################
def describe_portfolio() -> str:
    """
    Provides a textual summary of the portfolio by describing the Knock-Out certificates within it.

    Returns
    -------
    str
        A formatted string summarizing the portfolio's Knock-Out certificates with the following details
    """
    output = "*Current the portfolio includes the following certificates:*\n\n"

    isin_list = [f"ISIN: {product.isin}" for product in portfolio.ko_certificates]
    direction_list = [f"Type: {product.direction}" for product in portfolio.ko_certificates]
    last_leverage_list = [f"Last Leverage: {product.leverage_series.iloc[-1].item():.2f}x" for product in
                          portfolio.ko_certificates]
    base_price_change_list = [f"Base Price Change p.a.: {product.base_price_change_per_annum * 100:.2f} %" for product
                              in portfolio.ko_certificates]

    for isin, direction, last_leverage, base_price_change in zip(isin_list, direction_list, last_leverage_list,
                                                                 base_price_change_list):
        product_str = f"*{isin}*\n{direction}\n{last_leverage}\n{base_price_change}\n\n"
        output += product_str
    return output


def describe_agent():
    """
    Provides a description of the agent's current policy configuration based on its type.

    Returns
    -------
    str
        A detailed textual description of the agent's policy configuration if the agent is of type 'MultiProductAgent'.
        If the agent is not a 'MultiProductAgent', a message indicating that the agent is not manually defined and no details about the policy can be provided is returned.

    Notes
    -----
    - The description includes the agent's potential thresholds and corresponding actions (e.g., hold, sell, buy) based on the `env.potential_horizon_days`.
    - For 'MultiProductAgent', the behavior is described in terms of thresholds for"""
    output = "*Currently the agent is configured as follows:*\n\n"

    if not isinstance(agent, MultiProductAgent):
        return "Agent is not manually defined. I cannot elaborate on the RL policy."
    else:
        output += f"*Hold* if {env.potential_horizon_days}-day-potential is below threshold: ±{agent.abs_potential_threshold_steps[0] * 100} %\n"
        output += f"*Sell* linearly decreasing leverage if potential is between ±{agent.abs_potential_threshold_steps[0] * 100} % and ±{agent.abs_potential_threshold_steps[1] * 100} %\n"
        output += f"*Buy* linearly increasing leverage if potential is between ±{agent.abs_potential_threshold_steps[1] * 100} % and ±{agent.abs_potential_threshold_steps[2] * 100} %\n"
    return output


def describe_open_positions():
    """
    Provides a description of currently open positions within the environment.

    Iterates through the environment's open positions, if any exist, and formats them into a human-readable string. Includes details such as the financial instrument's ISIN, number of shares, leverage, price, direction, and percentage share of the portfolio. If no positions are open, the function returns a string indicating no open positions.

    Returns
    -------
    str
        A string describing the current open positions or indicating that none exist.
    """
    output = "*Currently there are the following open positions:*\n\n"

    if len(env.open_positions) == 0:
        return "Currently there are no open positions."
    else:  # iterate through entries and attach to output str
        for isin, (shares, leverage, price, direction, share_of_portfolio) in env.open_positions.iterrows():
            output += f"*{isin}*\nShares: {shares}\nLeverage: {leverage:.2f}x\nPrice: {price:.2f} €\nDirection: {direction}\nShare of Portfolio: {share_of_portfolio:.2f} %\n\n"
    return output


def describe_env_predictors() -> str:
    """
    Provides a description of the current predictors in the environment.

    Returns
    -------
    str
        A string representation of the current predictors in the environment.
    """
    return "*Currently the environment includes the following predictors:*\n\n" + str(env.predictor_instances)


def describe_workflows() -> str:
    """
    Describe the currently scheduled workflows and their execution times.

    This function iterates through a dictionary (`function_schedule`) that maps function names to their respective schedules. Each schedule contains execution time details in the order of Day, Weekday, Hour, and Minute. The function constructs a descriptive string providing the execution details for each workflow. It also converts numeric weekday values to their corresponding weekday names for better readability.

    Returns
    -------
    str
        A descriptive string listing all scheduled workflows along with their execution times, grouped by Day, Weekday, Hour, and Minute.
    """
    output = "*Currently the following workflows are scheduled:*\n"
    for func, execution_time in function_schedule.items():
        output += f"\n*{func}* to be executed on\n"
        for entry, label in zip(execution_time, ["Day", "Weekday", "Hour", "Minute"]):
            if entry is None: continue
            day_weekday_dict = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

            # allow for multiple entries:
            if not isinstance(entry, list): entry = [entry]
            output += f"{label}{'s' if len(entry) > 1 else ''}: "
            for int_entry in entry:
                if label == "Weekday": int_entry = day_weekday_dict[int_entry]
                output += f"{int_entry}, "
            # remove last comma and whitespace
            output = output[:-2]
            output += "\n"

    return output


def describe_predictor_presets() -> str:
    """
    Describes the predictor presets available.

    This function retrieves a description of the different preset types
    available in the predictor manager. Presets typically define
    pre-configured settings or strategies for the predictor,
    allowing for streamlined initialization and usage.

    Returns
    -------
    str
        A string description of the available predictor presets.
    """
    return pred_manager.describe_preset_types()


def describe_best_predictors(presets_to_include: [str] = ("c1", "c2", "d1", "d2", "d3")) -> str:
    """
    Describes the best predictors based on the given preset types.

    This function iterates over a list of preset predictor types and retrieves the best predictor for each type
    based on its validation hit rate. It generates a formatted string summarizing the best available predictors
    along with their validation hit rates.

    Parameters
    ----------
    presets_to_include : list of str, optional
        A list of preset predictor types to include in the description. The default values are
        ["c1", "c2", "d1", "d2", "d3"].

    Returns
    -------
    str
        A formatted string listing the best predictors for the specified preset types along with their validation
        hit rates.

    Raises
    ------
    IndexError
        If there are no predictors available for a specific preset type (handled internally and skipped
        for that type).
    """
    output = "*Currently the best available predictors are:*\n\n"
    for predictor_type in presets_to_include:
        try:
            best_val_hr = pred_manager.get_predictors_by_type_sorted(architecture='LSTM', return_instances=False,
                                                       preset_type=predictor_type)[0]['validation_hit_rate']
            output += f"*'{predictor_type}':* with validation hit rate *{best_val_hr*100:.2f} %*\n"
        except IndexError:  # no predictor of that type included
            continue

    return output


def describe_best_backtests(criterion: Literal['Mean', 'Median', 'SharpeRatio'] = 'Mean', k_best: int = 5,
                            not_older_than: pd.Timestamp = None) -> str:
    """
    Parameters
    ----------
    criterion : {'Mean', 'Median', 'SharpeRatio'}, default 'Mean'
        The metric to sort the backtests and determine the best configurations.

    k_best : int, default 5
        The number of top backtests to select based on the specified criterion.

    not_older_than : pandas.Timestamp, optional
        A date filter to include only backtests performed on or after the provided timestamp.

    Returns
    -------
    str
        A formatted string describing the top `k_best` backtests based on the specified criterion.
    """
    backtest_frame = pd.read_csv(filemgmt.most_recent_file(SAVED_BACKTESTS))

    # select only recent
    backtest_frame['date'] = pd.to_datetime(backtest_frame['date'])
    if not_older_than is not None:
        backtest_frame = backtest_frame.loc[backtest_frame['date'] >= not_older_than]

    # select best:
    best_tests = backtest_frame.sort_values(by=criterion, ascending=False).iloc[:k_best].reset_index()

    output_str = f"Currently the {k_best} *best backtests* {f'since {not_older_than} ' if not_older_than is not None else ''}are:\n\n"
    for ind, row in best_tests.iterrows():
        output_str += f"*{ind + 1}. best configuration:*\n"

        def if_present_describe(key: str) -> str:
            if key in row.keys():
                return f"- {key}: {row[key]}\n"
            else:
                return ""

        for key in ['date', 'Mean', 'StdDev', 'SharpeRatio', 'predictor_instances', 'abs_potential_threshold_steps',
                    'trading_quantity_per_leverage_factor', 'leverage_categories',
                    'sell_opposite_direction_if_no_cash', 'sell_all_opposite_products_if_no_cash']:
            output_str += if_present_describe(key)
        output_str += "\n"
    return output_str


###################### WORKFLOW FUNCTIONS ######################
# SATURDAYS
@timed_callback_decorator(callback=chatter)
@retry_decorator(on_error_callback=chatter)
def fine_tune_predictors(patience_tuple: [int] = (10, 20), presets_to_include: [str] = ("c1", "c2", "d1", "d2", "d3")):
    # reinitialise predictor manager with larger not_older_than_n_days
    temp_pred_manager = PredictorManager(data_manager=data_manager, initialisation_dir=SAVED_MODELS, recursive=True,
                                         not_older_than_n_days=60,  # models of last two month
                                         )

    chatter(f"I am going to *fine-tune the predictors* with presets {presets_to_include}. This may take a while.")
    # fine tune best pred of defined types:
    # todo: define custom_step_loss_weight_range based on best parametrisations
    for ind, patience in enumerate(patience_tuple):  # two runs with lower and higher patience
        chatter(f"Fine-tuning run {ind+1}/{len(patience_tuple)} with patience {patience}.")
        temp_pred_manager.fine_tune_predictors(architectures_to_finetune=['LSTM'],
                                               types_to_finetune=presets_to_include,
                                               finetune_working_directory=WORKING_DIR_PREDICTORS,
                                               train_epochs=200, early_stopping_patience=patience, )

    # reinitialise predictor manager with small not_older_than_n_days
    global pred_manager  # modify beyond function scope
    pred_manager = PredictorManager(data_manager=data_manager, initialisation_dir=SAVED_MODELS, recursive=True,
                                    not_older_than_n_days=10,
                                    # models of last week (and some more days if fine-tuning was delayed)
                                    )

    chatter("Fine-tuning runs finished!")
    chatter(describe_best_predictors(presets_to_include=presets_to_include))


# todo: 1ST -> predictor_parametrisation_loop() -> print_results -> then reinitialise_pred_manager
@timed_callback_decorator(callback=chatter)
@retry_decorator(on_error_callback=chatter)
def parametrize_predictors():
    pass

@timed_callback_decorator(callback=chatter)
@retry_decorator(on_error_callback=chatter)
def back_test_predictors(presets_to_consider=("b1", "b2", "c1", "c2", "d1", "d2", "d3"),
                         n_predictors_range: [int] = (1, 3),
                         architectures_to_consider: [str] = ("LSTM",),):
    """
    Back-tests a range of predictor combinations based on the given preset types and architectures.

    Parameters
    ----------
    presets_to_consider : tuple of str, default=("b1", "b2", "c1", "c2", "d1", "d2", "d3")
        List of preset identifiers to consider for back-testing.

    n_predictors_range : tuple of int, default=(1, 4)
        Range specifying the minimum and maximum number of predictors to combine for testing.

    architectures_to_consider : tuple of str, default=("LSTM",)
        Neural network architectures to consider for building predictor combinations.
    """
    # resulting combinations:
    combs_to_backtest = []
    for number_of_presets in range(n_predictors_range[0], n_predictors_range[1] + 1):
        for comb in combinations(presets_to_consider, r=number_of_presets):
            combs_to_backtest.append(comb)

    # derive corresponding predictor instances:
    predictor_list = []
    for arch in architectures_to_consider:
        for comb in combs_to_backtest:
            list_entry = []
            for preset in comb:
                try:
                    list_entry.append(pred_manager.get_predictors_by_type_sorted(architecture=arch, preset_type=preset,
                                                                                 return_instances=True, k_best=1)[0])
                except IndexError:
                    print(f"No {arch} predictor found for {preset} in {comb}. Skipping.")
            if len(list_entry) > 0: predictor_list.append(list_entry)

    # status message:
    chatter(f"Will now backtest {len(predictor_list)} combinations of predictors, each taking approx. 10 minutes to complete, hence estimating a total of {len(predictor_list)*10/60:.2f} hours.")

    print(predictor_list)
    env_parametrisation_loop(
            env_price_series=data_manager.env_interp_prices,
            env_price_sampling_rate_minutes=data_manager.env_sampling_rate_minutes,
            env_product_set=backtest_portfolio,
            backtest_database_dir=SAVED_BACKTESTS,
            # varying params:
            predictor_instances=predictor_list,

            # constant params:
            # 90 day expected return, sell -> buy -> highest leverage
            trading_quantity_per_leverage_factor=2.5,
            abs_potential_threshold_steps=(.0025, .04, .1),
            sell_opposite_direction_if_no_cash=True,
            sell_all_opposite_products_if_no_cash=True,
            include_open_leverage_category=False,
            precalculate_predictor_observations=True,
            leverage_categories=(1.0, 2.0, 3.0, 4.0, 5.0))

    chatter("Back-testing finished!")
    chatter(describe_best_backtests(not_older_than=pd.Timestamp.now() - pd.Timedelta(days=2)))

@timed_callback_decorator(callback=chatter)
@retry_decorator(on_error_callback=chatter)
def update_portfolio(issue_date_offset_days:int = 50) -> None:
    """
    Updates the portfolio with new issue dates from web and saves the adjusted portfolio to a CSV file.

    Parameters
    ----------
    issue_date_offset_days : int, optional
        Number of days subtracted from the current date to set as the issue date for the products.
    """
    global portfolio  # to use and later overwrite global var

    chatter("Fetching updated product data from boerse-frankfurt.")
    for product in portfolio.ko_certificates:
        product.update_product_details_from_scrape(use_as_2nd_base_price=False)  # update product details from boerse-fra
        product.enforce_base_price_increase_per_annum(abs_increase_pa=.03)
        product.issue_date = datetime.now() - pd.Timedelta(days=issue_date_offset_days)  # very short issue date

    # save adjusted portfolio:
    portfolio.save_to_csv(SAVED_PORTFOLIOS / filemgmt.file_title("Scraped Certificate Set", ".csv"))

    # save to CSV
    portfolio = KOCertificateSet.load_from_csv(
        file_path=filemgmt.most_recent_file(SAVED_PORTFOLIOS, ".csv", ["Scraped"]),
        underlying_price_series=data_manager.non_etf_env_interp_prices, )
    chatter("Successfully updated portfolio.")

@timed_callback_decorator(callback=chatter)
@retry_decorator(on_error_callback=chatter)
def update_env_from_scrape(add_missing_isins: bool = False) -> tuple[float, float, dict[str, float], dict[str, float]]:
    # set env where one more step can be executed:
    env.current_step = env.total_steps - 2
    #chatter(f"Environment now is at {env.current_step_timestamp} from {env.step_timestamp_list[-1]}")

    ### scrape and update portfolio data:
    driver = webinteraction.login_to_wikifolio(WIKIFOLIO_USER_KEY[0], WIKIFOLIO_USER_KEY[1])
    wf_cash, wf_value, wf_shares_p_isin, wf_price_p_isin = webinteraction.scrape_portfolio_holdings_from_wikifolio(
        driver)

    # update environment accordingly:
    env.cash = wf_cash  # update env cash
    for isin, shares in wf_shares_p_isin.items():  # update open positions
        env.shares_per_product[isin] = shares

    # derive price deviations:
    closest_to_current_prices = env.product_set.price_frame.loc[:pd.Timestamp.now()].iloc[-1]
    print(f"-------------------- Simulated product price deviations --------------------")
    for isin, price in wf_price_p_isin.items():
        if isin in closest_to_current_prices.index:
            if price is None: continue
            deviation = (closest_to_current_prices[isin] / price - 1) * 100
            print(
                f"{isin}:  {deviation:.2f}%\t\t(Env. price: {price:.2f},  \tWF price: {closest_to_current_prices[isin]:.2f})")

    # see whether all product_set isins are included in wikifolio:
    missing_isins = []
    for product in env.product_set.ko_certificates:
        if product.isin not in wf_shares_p_isin.keys():
            print(
                f"Product {product.isin} not found in WF portfolio. {'Will be added.' if add_missing_isins else 'Consider adding.'}")
            missing_isins.append(product.isin)
    if add_missing_isins: webinteraction.add_products_to_wikifolio(driver, missing_isins)

    return wf_cash, wf_value, wf_shares_p_isin, wf_price_p_isin


@timed_callback_decorator(callback=chatter)
@retry_decorator(on_error_callback=chatter)
def predict_and_trade(add_missing_isins: bool = False):
    ### scrape and update price data
    # update interpolated data with scraping:
    #data_manager.update(force_include_scraped=True)
    env.price_series = data_manager.env_interp_prices
    portfolio.update_all_price_series(data_manager.non_etf_env_interp_prices)

    # scrape data from wikifolio
    wf_cash, wf_value, wf_shares_p_isin, _ = update_env_from_scrape(add_missing_isins=add_missing_isins)

    # status message:
    chatter("Hey! I just fetched wikifolio's statistics and today's price data. Now inferring today's predictions.")
    chatter(f"Currently the environment is at {env.current_step_timestamp}.")

    ### infer new action:
    action, _ = agent.predict(env.current_observation)  # predict action
    env.plot_current_predictions(save_fig_directory=PREDICTION_PLOTS, hidden=False)  # save prediction plot
    trade_implementor = TradeImplementor()  # log class
    new_obs, _, done, truncated, info = env.step(action, track_portfolio_exposure=False,
                                                 start_new_episode_if_finished=False,
                                                 trade_implementation_callback=trade_implementor)  # step env

    # status messages:
    predictions_str_list = [f"{round(potential * 100, 2)} % in {round(horizon_minutes / 60 / 14)} days" for
                            potential, horizon_minutes in
                            zip(env.current_potential_estimates, env.observation_horizons_minutes)]
    prediction_str = "\n".join(predictions_str_list)  # will be included in next str:
    explanatory_string = f"Based on the *predictions*\n\n{prediction_str}\n\n-> an *average predicted potential of {env.current_avg_scaled_predicted_potential * 100:.2f} %* in {env.potential_horizon_days} days, the agent wants to *{info['Action']}*!"
    chatter.send_message(message=explanatory_string,  # include prediction plot
                         image_path=filemgmt.most_recent_file(PREDICTION_PLOTS, ".png", "Prediction Visualisation"))

    ### execute / tell how to execute action:
    # status messages:
    chatter(f"Currently we have {wf_cash:.2f} EUR cash in the wikifolio ({wf_cash / wf_value * 100:.2f}%).")
    trade_str_list = []  # describe recommended trades:
    for trade_dict in trade_implementor.action_log.values():
        if trade_dict["completed"]: continue  # only uncompleted trade
        trade_str_list.append(
            f"*{trade_dict['action']} {trade_dict['amount'].item() if isinstance(trade_dict['amount'], np.float16) else trade_dict['amount']} shares* of {trade_dict['isin']}. This is row {list(wf_shares_p_isin.keys()).index(trade_dict['isin'])} in wikifolio's trading desk.")
    chatter("The resulting recommended trades are:")  # send via chatbot
    for trade_str in trade_str_list:
        chatter(trade_str)
        print(trade_str)
    chatter(f"Now the environment is at {env.current_step_timestamp}.")


def tell_time():
    chatter(f"It is {datetime.now().hour}.{datetime.now().minute} and this is a test")


def update_env_predictors(types_to_include=("c2", "d2", "d3")) -> None:
    """
    Updates the environment's predictors based on the specified types.

    Parameters
    ----------
    types_to_include : tuple of str, optional
        A tuple of predictor types to include (e.g., "c2", "d2", "d3"). Default is ("c2", "d2", "d3").
    """
    preds_to_include = [pred_manager.get_predictors_by_type_sorted(architecture='LSTM',
                                                                   preset_type=p_type,
                                                                   return_instances=True,
                                                                   k_best=1)[0] for p_type in types_to_include]
    env.predictor_instances = preds_to_include


###################### WORKFLOW DEFINITION ######################
# day (of month), weekday, hour, minute
function_schedule = {predict_and_trade: [None, [0, 1, 2, 3, 4], 16, 20],  # 20 minutes offset, because 15-min delayed prices and +5min to prevent errors
                     update_portfolio: [None,[0, 1, 2, 3, 4], 15, 20],  # update portfolio details to prepare prediction
                     fine_tune_predictors: [None, 5, 13, 0],
                     back_test_predictors: [None, 6, 13, 0],
                     parametrize_predictors: [15, None, 17, 0],
                     tell_time: [None, None, [13, 15] , [15, 45]],
                     }


###################### PROCESS DEFINITIONS ######################
def responsive_workflow_process(shared_input_str, shared_output_str, chatbot_input_event, response_ready_event):
    """
    This function is designed to run continuously in a loop and must be executed within a multiprocess/multithread environment.
    The function handles scheduled execution of tasks based on a predefined schedule (`function_schedule`) and processes chatbot input requests in parallel.
    A synchronized and efficient mechanism is implemented for inter-process communication, using shared memory and event triggers.
    
    Parameters
    ----------
    shared_input_str : multiprocessing.Array
        Shared memory object for storing the input string from a chatbot interface.
        It is expected to be a byte array that supports inter-process communication.

    shared_output_str : multiprocessing.Array
        Shared memory object for storing the output string that will be sent to the chatbot.
        It is expected to be a byte array and will store the chatbot's response in a similarly encoded format.

    chatbot_input_event : multiprocessing.Event
        Event object used to signal when the chatbot input string is ready to be processed.

    response_ready_event : multiprocessing.Event
        Event object used to signal when the chatbot response has been generated 
        and is ready to be accessed by the interfacing process.

    Notes
    -----
    - Task scheduling allows execution based on day, weekday, hour, and minute, and prevents multiple executions at the same scheduled time.
    - Input in `shared_input_str` undergoes mapping for specific requests, and if unrecognized, it responds with potential options.
    - Proper byte encoding and padding mechanisms are used to ensure seamless shared memory operations.
    """
    ############# SCHEDULER THREAD ##############
    def execute_scheduled_functions():
        """
        Separate function definition to run the scheduler in another thread.
        """
        # sanity check, consecutive minutes are not allowed in the schedule of one function (because that is the smallest
        # time unit in which we check for multiple executions)
        for func, schedule in function_schedule.items():
            _, _ , _, minute = schedule
            if isinstance(minute, list):
                for entry in minute:
                    if entry + 1 in minute:
                        raise ValueError(f"Consecutive minutes are not allowed! Please amend the schedule of {func.__name__}.")

        was_executed = [False] * len(function_schedule.keys())  # bools to prevent multiple execution

        # run schedule checker
        while True:
            now = datetime.now()
            for func_ind, (func, schedule) in enumerate(function_schedule.items()):
                day, weekday, hour, minute = schedule

                execute = True
                # check schedule
                if day is not None:
                    if isinstance(day, list):
                        if now.day not in day or was_executed[func_ind]:
                            execute = False
                    elif now.day != day or was_executed[func_ind]:
                        execute = False

                if weekday is not None:
                    if isinstance(weekday, list):
                        if now.weekday() not in weekday or was_executed[func_ind]:
                            execute = False
                    elif now.weekday() != weekday or was_executed[func_ind]:
                        execute = False

                if hour is not None:
                    if isinstance(hour, list):
                        if now.hour not in hour or was_executed[func_ind]:
                            execute = False
                    elif now.hour != hour or was_executed[func_ind]:
                        execute = False

                if minute is not None:
                    if isinstance(minute, list):
                        if now.minute not in minute or was_executed[func_ind]:
                            execute = False
                    elif now.minute != minute or was_executed[func_ind]:
                        execute = False

                # execute function
                if execute:
                    func()
                    was_executed[func_ind] = True

                # reset was_executed for the next scheduled time
                reset = True
                # check only the smallest provided timescale, because that is enough reason to reset
                if minute is not None:
                    if isinstance(minute, list) and now.minute - 1 not in minute:
                        reset = False
                    elif not isinstance(minute, list) and now.minute - 1 != minute:
                        reset = False
                elif hour is not None:
                    if isinstance(hour, list) and now.hour - 1 not in hour:
                        reset = False
                    elif not isinstance(hour, list) and now.hour - 1 != hour:
                        reset = False
                elif weekday is not None:
                    if isinstance(weekday, list) and now.weekday() - 1 not in weekday:
                        reset = False
                    elif not isinstance(weekday, list) and now.weekday() - 1 != weekday:
                        reset = False
                elif day is not None:
                    if isinstance(day, list) and now.day - 1 not in day:
                        reset = False
                    elif not isinstance(day, list) and now.day - 1 != day:
                        reset = False

                # conduct reset if we are in the next scheduled time:
                if reset: was_executed[func_ind] = False
        #### end of scheduler function definition

    # run scheduler in another thread:
    schedule_thread = threading.Thread(target=execute_scheduled_functions,
                                       daemon=True)  # daemon leads to termination of thread if main process terminates
    schedule_thread.start()

    ############# CHATBOT THREAD ##############
    global last_chatbot_input
    last_chatbot_input = ""
    while True:
        ##### at each iteration, check whether the connected chatbot process needs to read out information:
        if chatbot_input_event.is_set():
            # read out shared_input_str value and remove 0-byte-padding
            input_str = shared_input_str.value.rstrip(b"\x00").decode("utf-8")

            # generate response:
            request_map = {
                "describe portfolio": describe_portfolio,
                "describe agent": describe_agent,
                "describe open positions": describe_open_positions,
                "describe environment predictors": describe_env_predictors,
                "describe workflows": describe_workflows,
                "describe predictor presets": describe_predictor_presets,
                "describe best predictors": describe_best_predictors,
                "describe best backtests": describe_best_backtests,
                "do update environment": update_env_from_scrape,
                "do update environment predictors": update_env_predictors,
                "do step environment": predict_and_trade,
                "do update portfolio": update_portfolio,
            }
            # commands:
            if input_str == last_chatbot_input:
                output = "You wrote the same query as last time. Please write something different first. This helps to prevent redundant executions."
            elif input_str.lower()[:2] == "do" and input_str.lower() in request_map:
                _ = request_map[input_str.lower()]()
                output = "Done!"
            # descriptions:
            elif input_str.lower() in request_map:
                output = str(request_map[input_str.lower()]())
            elif "describe " + input_str.lower().strip() in request_map:
                output = str(request_map["describe " + input_str.lower().strip()]())
            elif input_str == "":
                output = ""  # empty input -> empty response
            else:
                output = "*Possible inputs are:*\n\n" + "\n".join(request_map.keys())

            # reset last input
            last_chatbot_input = input_str

            # write in shared memory (properly encoded to bytes)
            max_length = len(shared_output_str)  # Determine the maximum length
            truncated_output = output[:max_length]  # prepare overflow
            shared_output_str.value = truncated_output.encode("utf-8").ljust(max_length, b'\x00')

            # send event trigger to continue other process:
            response_ready_event.set()
            chatbot_input_event.clear()  # clear input event
            shared_input_str.value = b"\x00" * max_length  # clear output str


def responsive_chatbot_process(shared_input_str, shared_output_str, chatbot_input_event, response_ready_event):
    print("Don't forget to turn on ngrok terminal process via\nngrok http 8000 --domain prompt-crayfish-cunning.ngrok-free.app")
    #os.system("../private/ngrok-server.sh")  # doesn't work

    # initialise app:
    app = create_app()
    logging.info("Flask app started")

    # important! provide custom response logic via callable function:
    def chatbot_request(input_str):
        """ This function safes the input_str in shared memory, waits for the other process to trigger an event, and then returns the output_str."""
        # Truncate input_str to fit shared_input_str size to avoid overflow
        max_length = len(shared_input_str)  # Determine the maximum length
        truncated_input = input_str[:max_length]

        # trigger listening of other process:
        shared_output_str.value = b"\x00" * max_length  # clear output str
        chatbot_input_event.set()

        # amend shared_input_str to fill entire space allocated for shared_input_str and pads it with zero bytes
        shared_input_str.value = truncated_input.encode("utf-8").ljust(max_length, b'\x00')

        # retrieve output and again stripping it of the 0-byte-padding:
        response_ready_event.wait()  # wait for response event
        output = shared_output_str.value.rstrip(b'\x00').decode("utf-8")

        response_ready_event.clear()  # clear response event
        return output

    app.config["CUSTOM_RESPONSE_FUNCTION"] = chatbot_request

    # run app (loop)
    app.run(host="0.0.0.0", port=8000)


###################### PROCESS EXECUTION ######################
if __name__ == "__main__":
    ### initialise shared memory for inter-process communication:
    str_len = 3000  # fixed for shared memory

    # create shared ctypes arrays for strings:
    shared_input_str = multiprocessing.Array(c_char, str_len)
    shared_output_str = multiprocessing.Array(c_char, str_len)

    # initialise with empty strings:
    shared_input_str.value = b"\x00" * str_len
    shared_output_str.value = b"\x00" * str_len

    # trigger_event for timing of response read-out:
    chatbot_input_event = multiprocessing.Event()
    response_ready_event = multiprocessing.Event()

    # trigger processes:
    p1 = multiprocessing.Process(target=responsive_workflow_process, args=(shared_input_str, shared_output_str, chatbot_input_event, response_ready_event), name="WorkflowProcess")
    p2 = multiprocessing.Process(target=responsive_chatbot_process, args=(shared_input_str, shared_output_str, chatbot_input_event, response_ready_event), name="ChatbotProcess")

    p1.start()
    p2.start()

    p1.join()
    p2.join()