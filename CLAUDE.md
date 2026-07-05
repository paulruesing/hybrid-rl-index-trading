# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hybrid RL Index Trading: a modular trading system combining supervised learning (LSTM/Transformer time-series prediction) and reinforcement learning (DQN/PPO) for algorithmic index trading on knock-out certificates. Predicts short-term price movements and uses RL agents to optimize trading decisions with dynamic leverage selection.

**Status**: Work in progress. Data pipeline, predictors, RL environment, manual agent, and chatbot are functional. RL agent training and production deployment are under development.

## Setup

```bash
conda env create -f environment.yml
conda activate rl_env
```

Key deps: Python 3.12, PyTorch, Gymnasium, Stable Baselines3, pandas, yfinance, alpha_vantage, selenium, Flask.

## Running

```bash
# Production: starts scheduler + chatbot in two processes
python src/workflow.py

# Development: use Jupyter notebooks in notebooks/
jupyter notebook
```

## Architecture

The system has a pipeline architecture orchestrated by `src/workflow.py`:

**Orchestration** (`src/workflow.py`): Two-process design using multiprocessing — a `WorkflowProcess` runs scheduled tasks (fine-tuning, backtesting, trading) via cron-like scheduling, and a `ChatbotProcess` runs a Flask server for WhatsApp/Email interaction. They communicate via `multiprocessing.Array` + `Event` for shared memory IPC.

**Data Pipeline** (`src/pipeline/preprocessing.py`): `StockPriceDataManager` downloads daily prices from Alpha Vantage, interpolates to multiple sampling rates (15min, 60min, 1d, 7d), and creates rolling window views as (X, Y) pairs for model training.

**Predictors** (`src/pipeline/predictors.py`): `PredictorManager` manages multiple LSTM and Transformer models with different time horizon presets (b1/b2 hourly, c1/c2 daily, d1/d2/d3 weekly). Models are fine-tuned periodically and kept for 30 days.

**RL Environment** (`src/pipeline/rl_environments.py`): `RLTradingEnv` is a Gymnasium environment where state = predictor outputs (price potentials) + cash + holdings, actions = buy/sell with leverage span selection (discrete action space), and reward = change in portfolio balance per step.

**Agents** (`src/pipeline/rl_agents.py`): `MultiProductAgent` implements threshold-based decision logic. Compatible with Gymnasium interface and designed for replacement by trained RL agents (DQN via stable-baselines3).

**Financial Products** (`src/pipeline/financial_products.py`): `KOCertificate` and `KOCertificateSet` model knock-out certificates with leverage computation, intrinsic value calculation, and web-scraped price data.

**Web Interaction** (`src/pipeline/web_interaction.py`): Selenium-based scraping from Wikifolio and Boerse Frankfurt for real-time certificate data and trade execution.

**Chatbot** (`src/pipeline/chatbot.py`): WhatsApp/Email interface with request mapping, backed by Flask app in `src/utils/chatbot_app/`.

### Data Flow (Trading Day)

1. `predict_and_trade()` triggers at 16:59 on weekdays
2. `StockPriceDataManager` downloads latest prices
3. `RLTradingEnv` updates price series and computes predicted potentials via `PredictorManager`
4. Agent selects action based on observation (potentials + cash + holdings)
5. Environment executes trade, computes reward
6. Chatbot sends recommendations + visualization

## Key Paths

- `private/` — API keys, credentials, runtime config (`env_configuration.txt`), schedule diary. **Never commit.**
- `data/saved_models/` — Trained LSTM/Transformer model checkpoints
- `data/interpolated_prices_dax/` — Resampled price data at various intervals
- `data/portfolios/` — Certificate portfolio CSVs
- `output/` — Backtest result plots and RL training logs

## Conventions

- Google-style docstrings on all major classes and methods (Parameters, Returns, Raises, Notes sections)
- Path constants defined at top of `workflow.py` using `pathlib.Path` (ROOT, DATA, DOWNLOADED_PRICES, etc.)
- Utility decorators in `src/utils/function_decorators.py` (timing, retry)
