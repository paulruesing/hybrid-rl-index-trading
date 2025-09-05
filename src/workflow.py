from src.pipeline.preprocessing import StockPriceDataManager
from src.pipeline.predictors import PredictorManager, predictor_parametrisation_loop
from src.pipeline.rl_environments import RLTradingEnv, env_parametrisation_loop, TradeImplementor
from src.pipeline.rl_agents import MultiProductAgent
from src.pipeline.financial_products import KOCertificate, KOCertificateSet
from src.pipeline.chatbot import WhatsAppChatbot
import src.pipeline.web_interaction as webinteraction
import src.utils.file_management as filemgmt

from datetime import datetime
import time
from typing import Union, Literal
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
                                not_older_than_n_days=12,  # only recently fine-tuned models
                                )
# todo: link pred_manager.describe_preset_types() to whatsapp

portfolio = KOCertificateSet.load_from_csv(
    file_path=filemgmt.most_recent_file(SAVED_PORTFOLIOS, ".csv", ["Scraped"]),
    underlying_price_series=data_manager.non_etf_env_interp_prices, )

backtest_portfolio = KOCertificateSet.load_from_csv(
        file_path=filemgmt.most_recent_file(SAVED_PORTFOLIOS, ".csv", ["Artificial"]),
        underlying_price_series=data_manager.non_etf_env_interp_prices, )

env = RLTradingEnv(price_series=data_manager.env_interp_prices,
                   price_sampling_rate_minutes=data_manager.env_sampling_rate_minutes,
                   predictor_instances=pred_manager.get_predictors_by_type_sorted(architecture='LSTM',
                                                                                  return_instances=True, k_best=3),
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
    output = "*Currently the agent is configured as follows:*\n\n"

    if not isinstance(agent, MultiProductAgent):
        return "Agent is not manually defined. I cannot elaborate on the RL policy."
    else:
        output += f"*Hold* if {env.potential_horizon_days}-day-potential is below threshold: ±{agent.abs_potential_threshold_steps[0] * 100} %\n"
        output += f"*Sell* linearly decreasing leverage if potential is between ±{agent.abs_potential_threshold_steps[0] * 100} % and ±{agent.abs_potential_threshold_steps[1] * 100} %\n"
        output += f"*Buy* linearly increasing leverage if potential is between ±{agent.abs_potential_threshold_steps[1] * 100} % and ±{agent.abs_potential_threshold_steps[2] * 100} %\n"
    return output


def describe_open_positions():
    output = "*Currently there are the following open positions:*\n\n"

    if len(env.open_positions) == 0:
        return "Currently there are no open positions."
    else:  # iterate through entries and attach to output str
        for isin, (shares, leverage, price, direction, share_of_portfolio) in env.open_positions.iterrows():
            output += f"*{isin}*\nShares: {shares}\nLeverage: {leverage:.2f}x\nPrice: {price:.2f} €\nDirection: {direction}\nShare of Portfolio: {share_of_portfolio:.2f} %\n\n"
    return output

def describe_predictors() -> str:
    return "*Currently the environment includes the following predictors:*\n\n" + str(env.predictor_instances)


def describe_workflows() -> str:
    output = "*Currently the following workflows are scheduled:*\n"
    for func, execution_time in function_schedule.items():
        output += f"\n*{func}* to be executed on\n"
        for entry, label in zip(execution_time, ["Day", "Weekday", "Hour", "Minute"]):
            if entry is None: continue
            output += f"{label}: {entry}\n"
    return output


###################### WORKFLOW FUNCTIONS ######################
# todo: SUNDAYS -> pred_manager.fine_tune_predictors() -> then reinitialise_pred_manager
def fine_tune_predictors():
    pass


# todo: 1ST -> predictor_parametrisation_loop() -> print_results -> then reinitialise_pred_manager


# todo: 15TH -> env_parametrisation_loop() -> print_results -> then eventually amend env


def predict_and_trade(add_missing_isins: bool = False):
    ### scrape and update price data
    # update interpolated data with scraping:
    data_manager.update(force_include_scraped=True)
    env.price_series = data_manager.env_interp_prices
    portfolio.update_all_price_series(data_manager.non_etf_env_interp_prices)

    # set env where one more step can be executed:
    env.current_step = env.total_steps - 2

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

    # status message:
    chatter("Hey! I just fetched wikifolio's statistics and today's price data. Now inferring today's predictions.")
    chatter(f"Currently the environment is at {env.current_step_timestamp}.")

    ### infer new action:
    action, _ = agent.predict(env.current_observation)  # predict action
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
    chatter(explanatory_string)

    ### execute / tell how to execute action:
    # status messages:
    chatter(f"Currently we have {wf_cash} EUR cash in the wikifolio ({wf_cash / wf_value * 100:.2f}%).")
    trade_str_list = []  # describe recommended trades:
    for trade_dict in trade_implementor.action_log.values():
        if trade_dict["completed"]: continue  # only uncompleted trade
        trade_str_list.append(
            f"*{trade_dict['action']} {trade_dict['amount'].item() if isinstance(trade_dict['amount'], np.float16) else trade_dict['amount']} shares* of {trade_dict['isin']}. This is row {list(wf_shares_p_isin.keys()).index(trade_dict['isin'])} in wikifolio's trading desk.")
    chatter("The resulting recommended trades are:")  # send via chatbot
    for trade_str in trade_str_list:
        chatter(trade_str)
    chatter(f"Now the environment is at {env.current_step_timestamp}.")


def test_at_12_min():
    chatter(f"It is {datetime.now().hour}.12 and this is a test")


###################### WORKFLOW DEFINITION ######################
# day (of month), weekday, hour, minute
function_schedule = {predict_and_trade: [None, None, 16, 00],
                     fine_tune_predictors: [None, 7, 10, 0],
                     test_at_12_min: [None, None, None, 12],
                     }


###################### PROCESS DEFINITIONS ######################
def responsive_workflow_process(shared_input_str, shared_output_str, chatbot_input_event, response_ready_event):
    was_executed = [False] * len(function_schedule.keys())  # bools to prevent multiple execution

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
                "describe predictors": describe_predictors,
                "describe workflows": describe_workflows,
            }
            if input_str.lower() in request_map:
                output = str(request_map[input_str.lower()]())
            else:
                output = "*Possible inputs are:*\n\n" + "\n".join(request_map.keys())

            # write in shared memory (properly encoded to bytes)
            max_length = len(shared_output_str)  # Determine the maximum length
            truncated_output = output[:max_length]  # prepare overflow
            shared_output_str.value = truncated_output.encode("utf-8").ljust(max_length, b'\x00')

            # send event trigger to continue other process:
            response_ready_event.set()
            chatbot_input_event.clear()  # clear input event
            shared_input_str.value = b"\x00" * max_length  # clear output str

        #### and check whether any scheduled function needs to be executed:
        now = datetime.now()
        for func_ind, (func, schedule) in enumerate(function_schedule.items()):
            if was_executed[func_ind]: continue
            day, weekday, hour, minute = schedule
            execute = True

            # check schedule
            if day is not None and now.day != day:
                execute = False
            if weekday is not None and now.weekday() != weekday:
                execute = False
            if hour is not None and now.hour != hour:
                execute = False
            if minute is not None and now.minute != minute:
                execute = False

            # execute function
            if execute:
                func(); was_executed[func_ind] = True

            reset = True
            # if schedule is just missed, reset was_executed:
            if day is not None and now.day - 1 != day:
                reset = False
            if weekday is not None and now.weekday() - 1 != weekday:
                reset = False
            if hour is not None and now.hour - 1 != hour:
                reset = False
            if minute is not None and now.minute - 1 != minute:
                reset = False
            if reset: was_executed[func_ind] = False


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