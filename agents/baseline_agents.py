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



class LinearDiscountAgent:
    """
    A baseline heuristic agent that starts at a high price and decreases it daily.
    Default: starts at ₹6000 on Day 1, and drops ₹200/day.
    Maps the target price to the nearest available price level in the environment.
    """

    def __init__(
        self,
        start_price: float = 6000.0,
        daily_drop: float = 200.0,
        max_days: int = 30,
        prices: list | None = None,
    ) -> None:
        self.start_price = start_price
        self.daily_drop = daily_drop
        self.max_days = max_days
        self.prices = prices if prices is not None else list(range(500, 10001, 500))

    def __call__(self, obs: np.ndarray) -> int:
        return self.act(obs)

    def act(self, obs: np.ndarray) -> int:
        # obs is [remaining_inventory, days_until_departure]
        days_left = obs[1]
        days_passed = self.max_days - days_left
        target_price = self.start_price - (self.daily_drop * days_passed)

        # Find closest price index
        action_index = int(np.argmin([abs(p - target_price) for p in self.prices]))
        return action_index


class RandomAgent:
    """
    A baseline agent that selects a random action (price level index).
    """

    def __init__(self, action_space_size: int = 20) -> None:
        self.action_space_size = action_space_size

    def __call__(self, obs: np.ndarray) -> int:
        return self.act(obs)

    def act(self, obs: np.ndarray) -> int:
        return int(np.random.randint(0, self.action_space_size))
