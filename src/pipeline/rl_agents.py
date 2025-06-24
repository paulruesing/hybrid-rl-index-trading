from typing import Literal
import numpy as np

class MultiProductAgent:
    """
    Manual trading agent callable with RLTradingEnv observations.

    This is a base class for agents that make decisions based on multi-input observation types. The default
    decision logic uses potential estimates to determine an action via preset thresholds. Users may override
    the `get_decision` method to implement custom decision strategies.
    """
    def __init__(self,
                 observation_types: [Literal['potential', 'tendency', 'cash', 'holding']],
                 n_leverage_categories: int,
                 include_open_leverage_category: bool = False,
                 abs_potential_treshold_steps: (float, float, float) = (.0025, .0075, .015),
                 ):
        """
        Initialize a MultiProductAgent instance.

        Parameters
        ----------
        observation_types : list of {'potential', 'tendency', 'cash', 'holding'}
            The types of observations the agent expects in a fixed order, e.g.
            ['potential', 'cash', 'holding'].
        n_leverage_categories : int
            Number of leverage categories to consider when determining actions.
        include_open_leverage_category : bool, default False
            Whether to include an open-ended leverage category for very high/low potentials.
        abs_potential_treshold_steps : tuple of float, default (.0025, .0075, .015)
            Threshold steps that define zones of potential:
            - Below the first threshold: no action.
            - Between first and second: opposite direction products are sold.
            - Between second and third: products are bought in the estimated direction.
            Thresholds are linearly scaled according to the number of leverage categories.
        """
        self.observation_types = observation_types

        # initialise thresholds
        self.abs_potential_treshold_steps = abs_potential_treshold_steps
        if not include_open_leverage_category: n_leverage_categories -= 1  # remove (last_leverage, inf) category
        buy_long_thresholds = np.linspace(abs_potential_treshold_steps[1], abs_potential_treshold_steps[2],
                                          n_leverage_categories)  # within this range longs are bought
        sell_long_thresholds = (np.linspace(abs_potential_treshold_steps[0], abs_potential_treshold_steps[1],
                                            n_leverage_categories, endpoint=False) * -1)[
                               ::-1]  # within this range longs are sold, reversed because first highest leverages are sold
        buy_short_thresholds = (buy_long_thresholds * -1)[::-1]  # and vice versa
        sell_short_thresholds = (sell_long_thresholds * -1)[::-1]

        # intialise actions:
        buy_long_actions = list(range(1, 4 * n_leverage_categories + 1, 4))
        sell_long_actions = list(range(2, 4 * n_leverage_categories + 1, 4))
        buy_short_actions = list(range(3, 4 * n_leverage_categories + 1, 4))
        sell_short_actions = list(range(4, 4 * n_leverage_categories + 1, 4))

        # join:
        self.lower_thresholds = np.concat(
            [[-np.inf], buy_short_thresholds, sell_long_thresholds, sell_short_thresholds, buy_long_thresholds])
        self.actions = np.concat([buy_short_actions, sell_long_actions, [0], sell_short_actions, buy_long_actions])

    def describe(self) -> str:
        """
        Describe the threshold ranges and associated actions.

        Returns
        -------
        str
            Human-readable explanation of threshold-action mappings.
        """
        intro_str = "------------------- KOCertificate Instance -------------------\n\n"
        lines = ["Threshold-action mapping:"]
        for i, lower in enumerate(self.lower_thresholds):
            upper = self.lower_thresholds[i + 1] if i + 1 < len(self.lower_thresholds) else np.inf
            action = self.actions[i]
            lines.append(f"  {lower:>8.5f} < potential < {upper:>8.5f}  -->  action {action}")
        threshold_explanation_str = "\n".join(lines)
        return intro_str + threshold_explanation_str

    #### str operators ####
    def __repr__(self) -> str:
        """
        Official string representation of the agent.

        Returns
        -------
        str
            Developer-friendly string summarizing configuration and thresholds.
        """
        return self.describe()

    def __str__(self) -> str:
        """
        Informal string representation of the agent.

        Returns
        -------
        str
            Readable summary for end-users.
        """
        return self.describe()

    def __call__(self, observation: np.ndarray):
        """
        Make the agent callable.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector matching the order and type specified in `observation_types`.

        Returns
        -------
        tuple of (int, float)
            The selected action and a placeholder (NaN) for compatibility with DQN agents.
        """
        return self.predict(observation)

    def get_decision(self, potential_estimates: [float], tendency_estimates: [int] = None, cash: float = None,
                     holding: float = None) -> int:
        """
        Make a decision based on provided observations.

        Parameters
        ----------
        potential_estimates : list of float
            Estimates of market potential per asset. Positive indicates long signal, negative short.
        tendency_estimates : list of int, optional
            Optional binary indicators of recent direction; 1 for upward, 0 for downward.
        cash : float, optional
            Available cash for trading.
        holding : float, optional
            Current position holding.

        Returns
        -------
        int
            Encoded action index indicating the type and direction of trade.
        """
        # average estimates: (todo: consider weighting here)
        avg_potential = np.mean(potential_estimates)
        avg_tendency = np.mean(tendency_estimates) if tendency_estimates is not None else 0

        # select action based on thresholds:
        for ind, lower in enumerate(self.lower_thresholds):
            # derive upper threshold (or set to inf if last ind)
            upper = self.lower_thresholds[ind + 1] if ind + 1 < len(self.lower_thresholds) else np.inf
            # check for threshold and return respective action if met
            if lower < avg_potential < upper:
                return self.actions[ind]

    def predict(self, observation: np.ndarray) -> (int, None):
        """
        Compute the action to take based on the current observation.

        Parameters
        ----------
        observation : np.ndarray
            A vector of features corresponding to the expected `observation_types`.

        Returns
        -------
        tuple of (int, float)
            Action index and NaN placeholder for compatibility with typical RL agents.
        """
        # sanity check
        if len(self.observation_types) != len(observation): raise ValueError(
            "observation_types have to be specified (precisely) for each entry in observation array.")

        # treat observations by type:
        potential_estimates = np.array([], dtype=np.float64)
        tendency_estimates = np.array([], dtype=np.float64)  # 1 should be up, 0 down
        cash = holding = np.nan
        for type, obs in zip(self.observation_types, observation):
            if type == 'potential':
                potential_estimates = np.append(potential_estimates, [obs])
            elif type == 'tendency':
                tendency_estimates = np.append(tendency_estimates, [obs])
            elif type == 'cash':
                cash = obs
            elif type == 'holding':
                holding = obs

        return self.get_decision(potential_estimates, tendency_estimates, cash,
                                 holding), np.nan  # return NaN at second pos because commonly DQN agents return also next state