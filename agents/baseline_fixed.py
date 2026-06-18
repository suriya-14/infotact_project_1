import numpy as np


class FixedPriceAgent:
    """
    A baseline heuristic agent that always selects a fixed price index.
    In the default AirlinePricingEnv, action 7 corresponds to a price of ₹4000.
    """

    def __init__(self, action_index: int = 7) -> None:
        self.action_index = action_index

    def __call__(self, obs: np.ndarray) -> int:
        """
        Policy function signature: takes observation and returns action index.
        """
        return self.action_index

    def act(self, obs: np.ndarray) -> int:
        """
        Alternative method to select action.
        """
        return self.action_index
