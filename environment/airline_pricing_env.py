import gymnasium as gym
import numpy as np


class AirlinePricingEnv(gym.Env):
    """
    A custom Gymnasium environment simulating a dynamic pricing booking market
    for finite, perishable inventory (e.g., airline seats or hotel rooms).

    State representation:
        [remaining_inventory, days_until_departure]

    Action space:
        Discrete price level index mapping to a pricing menu (e.g., [2000, 3000, 4000, 5000, 6000])

    Reward:
        Revenue generated in the current step (price * items sold)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    DEFAULT_PRICES = [2000, 3000, 4000, 5000, 6000]
    __slots__ = (
        "max_inventory", "max_days", "prices", "base_demand",
        "price_sensitivity", "urgency_factor_slope", "render_mode",
        "action_space", "observation_space", "inventory", "days_left",
        "total_revenue", "_obs_buffer",
    )

    def __init__(
        self,
        max_inventory: int = 50,
        max_days: int = 30,
        prices: list | None = None,
        base_demand: float = 10.0,
        price_sensitivity: float = -0.0003,
        urgency_factor_slope: float = 0.05,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        valid_modes = self.metadata["render_modes"] + [None]
        if render_mode not in valid_modes:
            raise ValueError(f"Invalid render_mode: {render_mode}. Must be one of {valid_modes}")

        self.max_inventory = max_inventory
        self.max_days = max_days
        self.prices = prices if prices is not None else list(self.DEFAULT_PRICES)
        self.base_demand = base_demand
        self.price_sensitivity = price_sensitivity
        self.urgency_factor_slope = urgency_factor_slope
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(self.prices))

        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([float(max_inventory), float(max_days)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._obs_buffer = np.empty(2, dtype=np.float32)
        self.inventory = 0
        self.days_left = 0
        self.total_revenue = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days
        self.total_revenue = 0.0

        self._obs_buffer[0] = self.inventory
        self._obs_buffer[1] = self.days_left
        return self._obs_buffer, {}

    def get_demand(self, price: float) -> int:
        days_passed = self.max_days - self.days_left
        urgency_factor = 1.0 + (self.urgency_factor_slope * days_passed)

        expected_bookings = (
            self.base_demand
            * np.exp(self.price_sensitivity * price)
            * urgency_factor
        )
        return self.np_random.poisson(expected_bookings)

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in {self.action_space}")

        price = self.prices[action]
        raw_demand = self.get_demand(price)
        sold = raw_demand if raw_demand < self.inventory else self.inventory
        revenue = float(price * sold)

        self.inventory -= sold
        self.days_left -= 1
        self.total_revenue += revenue

        terminated = (self.inventory <= 0) or (self.days_left <= 0)

        self._obs_buffer[0] = self.inventory
        self._obs_buffer[1] = self.days_left

        info = {
            "total_revenue": self.total_revenue,
            "sold": sold,
            "demand": raw_demand,
            "price": price,
        }

        if self.render_mode == "human":
            self.render()

        return self._obs_buffer, revenue, terminated, False, info

    def render(self):
        message = (
            f"Days Left: {self.days_left:2d} | "
            f"Inventory: {self.inventory:2d}/{self.max_inventory:2d} | "
            f"Episodic Revenue: {self.total_revenue:.2f}"
        )
        if self.render_mode == "human":
            print(message)
        elif self.render_mode == "ansi":
            return message

    def close(self) -> None:
        self.inventory = 0
        self.days_left = 0
        self.total_revenue = 0.0
