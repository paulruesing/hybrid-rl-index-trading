# Hybrid RL Index Trading

This repository aims to leverage **supervised learning (based on time-series and language analysis)** and **reinforcement
learning (for decision-making)** for algorithmic **index trading**. This combination (**hybrid RL**) is expected to harvest
synergies and be more effective than either single architecture alone.

The code was developed in my free time and is a **WORK IN PROGRESS**.


## 1. What is the goal?
To develop a **modular and robust trading system** that:

- **Predicts future price movements** using LSTM-based supervised learning models trained on historical price data and (optionally) language-derived sentiment.
- Makes **real-time trading decisions** via a reinforcement learning agent interacting with a custom OpenAI Gym environment.
- Executes trades on **index-based derivatives** (e.g., knock-out certificates) based on predicted market direction and calibrated risk appetite (via leverage).

- The ultimate objective is to create a system that learns to profit consistently in volatile market conditions by combining **foresight** (via prediction) with **adaptability** (via RL-based strategy refinement).

## 2. How is it done?
- **Data Preparation**: Historical price data is loaded from CSV files, resampled to fixed intervals (e.g., 1 minute), and structured into input windows suitable for sequential models.
- **Supervised Learning**: A Long Short-Term Memory (LSTM) neural network is trained to forecast short-term price changes or directional movements.
- **Trading Environment**: A custom RL environment is built using OpenAI Gym to simulate realistic trading with knock-out certificates. The action space is defined by selecting leverage levels from predefined categories.
- **Reinforcement Learning**: An RL agent (e.g., PPO or DQN) learns to select actions based on both the current market state and predicted future trends. The environment incorporates reward functions that consider returns, risk exposure, and trading costs.
- **Hybrid Integration**: The LSTM predictor is used to enrich the state input to the RL agent, thereby improving the quality of decision-making under uncertainty.

## 3. Why this approach?
- **Noise Robustness**: Market prices are inherently noisy. Pure RL systems often struggle in such settings without guidance. By integrating LSTM-based forecasts, the RL agent can make more informed decisions.
- **Risk-Control via Leverage Selection**: Using leverage as the primary action variable aligns the system with realistic product constraints (e.g., knock-outs), while offering a controllable risk-return spectrum.
- **Modularity**: Separating the prediction and decision modules allows independent development and tuning, promoting scalability and experimentation.
- **Realism**: The setup reflects actual retail products and market mechanics, making the model more applicable to real-world deployment or analysis.

## 4. What's next?
- Implement support for **sentiment-based feature extraction** from financial news or social media to complement price-based signals
- Extend the ensemble model by **market regime classifiers** and **additional prediction architectures** (transformers)
- Move from manual agents to **RL-trained agents**
- Extend the RL agent’s capabilities to include **position sizing and timing**
- Integrate **transaction costs, spread, and slippage models** for more realistic evaluation
- Perform **hyperparameter tuning** and ablation studies to evaluate the contribution of each component (prediction, reward shaping, action space granularity)
- Add **logging and visualization dashboards** for real-time simulation tracking

## 5. Notebooks structure
- *src/*: source code directory containing classes and methods
- *notebooks/*: jupyter notebooks demonstrating the workflow and necessary for development
- *literature/*: a selection of papers explicating some theoretical underlinings
- *data/*: input data and saved models

## 6. How to run?
### 6.1. Required Modules
It is recommended to install all required modules by creating a conda environment through running
`conda env create -f environment.yml`
in terminal in the project directory.

### 6.2. Recommendations
Usage is extensively demonstrated in the notebooks, and it is advised to follow such procedure when implementing.

## 7. Other Important Information
### 7.1. Authors and Acknowledgment
Paul Rüsing - pr@paulruesing.de - single and main contributor

### 5.2. License
The project is licensed under the MIT license. To view a copy of this license, see [LICENSE](https://github.com/paulruesing/lrp-xai-pytorch?tab=MIT-1-ov-file).
