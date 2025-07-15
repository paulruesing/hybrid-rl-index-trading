import src.utils.file_management as filemgmt
import src.pipeline.preprocessing as preprocessing
from src.pipeline.predictors import LSTMPredictor
from src.pipeline.rl_environments import RLTradingEnv
from src.pipeline.rl_agents import MultiProductAgent
from src.pipeline.financial_products import KOCertificate, KOCertificateSet

from itertools import product
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """ Main function. Executed upon terminal run. """
    # necessary files:
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    INTERPOLATED_PRICES = DATA / "interpolated_prices"
    LOG_FILES = ROOT / "output" / "rl_training_logs"

    SAVED_MODELS = DATA / "saved_models"
    SAVED_B1_PREDICTOR = SAVED_MODELS / "predictor_b1" / "2025-05-22 10_03_41 LSTM Model RW68 FH14 Layers3 Size256 TrainL0.10641186928842217 ValL1.9110333621501923 TrainHR0.4860040545463562 ValHR0.5328466892242432.pt"
    SAVED_B3_PREDICTOR = SAVED_MODELS / "predictor_b3" / "2025-05-22 11_47_10 LSTM Model RW40 FH5 Layers3 Size128 TrainL0.36055382899940014 ValL1.9578219056129456 TrainHR0.547325074672699 ValHR0.529629647731781.pt"
    SAVED_C1_PREDICTOR = SAVED_MODELS / "predictor_c1" / "2025-05-22 11_27_18 LSTM Model RW24 FH3 Layers4 Size128 TrainL0.29682125337421894 ValL0.4320639967918396 TrainHR0.5574468374252319 ValHR0.5660377144813538.pt"

    ENV_PRICE_FILE = filemgmt.most_recent_file(INTERPOLATED_PRICES, '.csv', '15min')  # filemgmt.most_recent_file(INTERPOLATED_PRICES, '.csv', '15min')
    B1_PRICE_FILE = filemgmt.most_recent_file(INTERPOLATED_PRICES, '.csv', '60min')
    B3_PRICE_FILE = filemgmt.most_recent_file(INTERPOLATED_PRICES, '.csv', '1d')
    C1_PRICE_FILE = filemgmt.most_recent_file(INTERPOLATED_PRICES, '.csv', '7d')
    MEAN_ETF_INDEX_RATIO = 543.3079783216567  # because files are etf prices

    # initialise portfolio:
    portfolio = KOCertificateSet(
        underlying_price_series=preprocessing.read_price_csv(ENV_PRICE_FILE) * MEAN_ETF_INDEX_RATIO,
        base_price_inference_timestamps=["2020-07-06 10:00:00", "2022-07-05 10:00:00", "2024-01-05 10:00:00"],
        # "2023-07-05 10:00:00", "2024-07-05 10:00:00", "2025-03-05 10:00:00"],
        n_products_per_direction=15,
        lowest_leverage=1.0, highest_leverage=10.0)

    # add further products:
    portfolio.future_product_instances += portfolio.initialise_products_from_leverage(
        underlying_price_series=preprocessing.read_price_csv(ENV_PRICE_FILE) * MEAN_ETF_INDEX_RATIO,
        base_price_inference_timestamps=["2017-12-08 10:00:00", "2018-07-05 10:00:00"],
        n_products_per_direction=15, issue_date="2017-01-01 10:00:00",
        lowest_leverage=1.0, highest_leverage=10.0)
    portfolio.future_product_instances += portfolio.initialise_products_from_leverage(
        underlying_price_series=preprocessing.read_price_csv(ENV_PRICE_FILE) * MEAN_ETF_INDEX_RATIO,
        base_price_inference_timestamps=["2021-07-06 10:00:00", "2023-07-05 10:00:00", "2024-07-05 10:00:00",
                                         "2025-03-05 10:00:00"],
        n_products_per_direction=15, issue_date="2020-06-01 10:00:00",
        lowest_leverage=1.0, highest_leverage=10.0)
    portfolio.future_product_instances += portfolio.initialise_products_from_leverage(
        underlying_price_series=preprocessing.read_price_csv(ENV_PRICE_FILE) * MEAN_ETF_INDEX_RATIO,
        base_price_inference_timestamps=["2023-05-05 10:00:00", "2024-07-05 10:00:00", "2025-03-05 10:00:00"],
        n_products_per_direction=15, issue_date="2022-12-01 10:00:00",
        lowest_leverage=1.0, highest_leverage=10.0)
    portfolio.future_product_instances += portfolio.initialise_products_from_leverage(
        underlying_price_series=preprocessing.read_price_csv(ENV_PRICE_FILE) * MEAN_ETF_INDEX_RATIO,
        base_price_inference_timestamps=["2025-03-05 10:00:00"],
        n_products_per_direction=15, issue_date="2024-03-03 10:00:00",
        lowest_leverage=1.0, highest_leverage=10.0)
    portfolio.future_product_instances += portfolio.initialise_products_from_leverage(
        underlying_price_series=preprocessing.read_price_csv(ENV_PRICE_FILE) * MEAN_ETF_INDEX_RATIO,
        base_price_inference_timestamps=["2025-03-07 12:00:00"],
        n_products_per_direction=15, issue_date="2024-03-03 10:00:00",
        lowest_leverage=1.0, highest_leverage=10.0)

    # check leverage availability:
    if False:
        print('Long availabilities:')
        portfolio.get_leverage_availability("long", hour_minute_to_check=(16, 0))
        print('Short availabilities:')
        portfolio.get_leverage_availability("short", hour_minute_to_check=(16, 0))

    # initialise predictors:
    b1_predictor = LSTMPredictor(model_load_file_path=SAVED_B1_PREDICTOR,
                                 # based on this, the predictor infers the model's properties
                                 price_csv_path=B1_PRICE_FILE,
                                 daily_prediction_hour=16,
                                 # these are necessary data properties which still need to be set
                                 predict_before_daily_prediction_hour=True,
                                 rolling_window_size=14 * 4 + 8,
                                 # 1 week hindsight (4 days a 14 hours and 1 a 8 hours until 16.00)
                                 forecast_horizon=14,  # 1 day a 14 hours ahead
                                 sampling_rate_minutes=60)  # hourly
    b3_predictor = LSTMPredictor(model_load_file_path=SAVED_B3_PREDICTOR,
                                 # based on this, the predictor infers the model's properties
                                 price_csv_path=B3_PRICE_FILE,
                                 daily_prediction_hour=16,
                                 # these are necessary data properties which still need to be set
                                 rolling_window_size=2 * 4 * 5,  # 2 months a 4 weeks a 5 days hindsight
                                 forecast_horizon=5,  # 1 week a 5 days ahead
                                 sampling_rate_minutes=60 * 14,  # one day from 8 to 22
                                 )
    c1_predictor = LSTMPredictor(model_load_file_path=SAVED_C1_PREDICTOR,
                                 # based on this, the predictor infers the model's properties
                                 price_csv_path=C1_PRICE_FILE,
                                 daily_prediction_hour=16,
                                 # these are necessary data properties which still need to be set
                                 rolling_window_size=6 * 4,  # 6 months a 4 weeks hindsight
                                 forecast_horizon=3,  # 3 weeks ahead
                                 sampling_rate_minutes=5 * 14 * 60,  # 1 week a 5 days from 8 to 22
                                 )

    # initialise environment:
    self = RLTradingEnv(ENV_PRICE_FILE,
                        price_sampling_rate_minutes=15,
                        predictor_instances=(b3_predictor, c1_predictor),
                        product_set=portfolio, trading_quantity_per_leverage_factor=.5,
                        include_open_leverage_category=False,
                        leverage_categories=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0))  # , 5.5, 6.0, 6.5, 7.0))

    # initialise agent:
    agent = MultiProductAgent(observation_types=self.observation_types,
                              observation_horizons_minute=self.observation_horizons_minutes,
                              n_leverage_categories=len(self.leverage_categories),
                              abs_potential_treshold_steps=(.04, .15, .5),
                              include_open_leverage_category=self.include_open_leverage_category,
                              potential_treshold_horizon_days=self.potential_horizon_days,
                              )

    # prepare environment:
    obs = self.reset()  # reset episode and fetch first observation
    self.verbose = False
    done = False  # episode termination criterion
    log = pd.DataFrame()

    # run agent:
    for ind in tqdm(range(self.current_step, self.total_steps)):
        action, _ = agent.predict(obs)  # infer action
        obs, _, done, info = self.step(action,
                                       track_portfolio_exposure=(ind % 10 == 0),  # track portfolio exposure every 10 steps
                                       )  # retrieve new observation and info

        if done: break  # check whether episode is finished
        log = pd.concat([log, pd.DataFrame(info, index=[info['Step']])])  # log info

    # calculate benchmark:
    benchmark = self.starting_cash / self.price_series.loc[log['Time'].iloc[0]] * self.price_series.loc[
        log['Time']]  # construct benchmark return if HODL
    log.set_index('Time', inplace=True)
    log['Benchmark'] = benchmark
    # save log file:
    log.to_csv(LOG_FILES / filemgmt.file_title("Agent Training Log", ".csv"))

    # plot portfolio performance:
    val_df = log.loc[:, [f'Avg. Expected Potential / {self.potential_horizon_days}d', 'Total Exposure']]
    val_df['Policy'] = log['Total'] / log['Benchmark'].iloc[0]
    val_df['Benchmark'] = log['Benchmark'] / log['Benchmark'].iloc[0]
    fig, (return_ax, exposure_ax) = plt.subplots(2, 1, figsize=(14, 14))
    potential_ax = exposure_ax.twinx()
    dates = pd.to_datetime(log.index)

    # portfolio performance:
    return_ax.plot(dates, val_df['Policy'], color='orange', label='Policy')
    return_ax.plot(dates, val_df['Benchmark'], color='blue', label='Benchmark')
    return_ax.set_title('Normalized Policy vs. Benchmark')
    return_ax.set_xlabel('Date')
    return_ax.set_ylabel('Normalized Balance')
    return_ax.legend()
    return_ax.grid(True)

    # include potential thresholds:
    if isinstance(agent, MultiProductAgent):
        for ind, (threshold, sign) in enumerate(product(agent.abs_potential_treshold_steps, [-1, 1])):
            potential_ax.axhline(y=threshold * sign * 100, color='purple',
                                 linestyle=':', alpha=.3,
                                 label='Action Thresholds' if ind == 0 else '_',
                                 # create legend entry only for first line
                                 )

    # avg potential:
    potential_ax.plot(dates, val_df['Avg. Expected Potential'] * 100, color='black', linestyle='dashed',
                      label='Avg. Expected Potential', alpha=.5)

    # portfolio exposure:
    pos_leverage = np.where(val_df['Total Exposure'] >= 0, val_df['Total Exposure'], np.nan)
    neg_leverage = np.where(val_df['Total Exposure'] < 0, val_df['Total Exposure'], np.nan)
    exposure_ax.plot(dates, pos_leverage, color='green', label='Positive Exposure')
    exposure_ax.plot(dates, neg_leverage, color='red', label='Negative Exposure')

    # formatting:
    exposure_ax.set_xlabel('Date')
    exposure_ax.legend(loc='upper left');
    potential_ax.legend(loc='lower right')
    exposure_ax.set_ylabel('Total Exposure');
    potential_ax.set_ylabel('Expected Forward Potential [%]')
    exposure_ax.set_ylim([-5, 5]);
    potential_ax.set_ylim([-5, 5])
    exposure_ax.set_title('Portfolio Exposure and Predicted Potential')
    exposure_ax.grid(True)

    plt.show()
