import gymnasium as gym
import numpy as np


class AirlinePricingEnv(gym.Env):
    """
    A custom Gymnasium environment simulating a dynamic pricing booking market
    for finite, perishable inventory (e.g., airline seats or hotel rooms).

    State representation:
        [remaining_inventory, days_until_departure]

    Action space:
        Discrete price level index mapping to a pricing menu
        (e.g., [500, 1000, ..., 10000] in Rs 500 increments)

    Reward:
        Revenue generated in the current step (price * items sold)

    Demand features:
        - Customer segments with per-segment price elasticity
        - S-curve urgency (slow start, rapid mid-horizon rise, plateau)
        - Inventory scarcity boost (FOMO)
        - Market regime switches (peak / normal / off-peak)
        - Log-normal market noise
        - Random booking bursts (group bookings)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    DEFAULT_PRICES = list(range(500, 10001, 500))
    __slots__ = (
        "max_inventory", "max_days", "prices", "base_demand",
        "customer_segments",
        "urgency_amplitude", "urgency_steepness", "urgency_midpoint",
        "scarcity_sensitivity",
        "demand_regimes", "regime_change_prob",
        "market_noise_scale",
        "burst_probability", "burst_min_size", "burst_max_size",
        "render_mode",
        "action_space", "observation_space", "inventory", "days_left",
        "total_revenue", "current_regime",
        "_obs_buffer", "_burst",
    )

    def __init__(
        self,
        max_inventory: int = 50,
        max_days: int = 30,
        prices: list | None = None,
        base_demand: float = 10.0,
        customer_segments: list[dict] | None = None,
        urgency_amplitude: float = 1.0,
        urgency_steepness: float = 8.0,
        urgency_midpoint: float = 0.5,
        scarcity_sensitivity: float = 0.3,
        demand_regimes: dict | None = None,
        regime_change_prob: float = 0.05,
        market_noise_scale: float = 0.1,
        burst_probability: float = 0.05,
        burst_min_size: int = 5,
        burst_max_size: int = 15,
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

        if customer_segments is None:
            customer_segments = [
                {"name": "leisure", "weight": 0.7, "price_sensitivity": -0.0004},
                {"name": "business", "weight": 0.3, "price_sensitivity": -0.0001},
            ]
        total_weight = sum(s["weight"] for s in customer_segments)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Segment weights must sum to 1.0, got {total_weight}")
        self.customer_segments = customer_segments

        self.urgency_amplitude = urgency_amplitude
        self.urgency_steepness = urgency_steepness
        self.urgency_midpoint = urgency_midpoint
        self.scarcity_sensitivity = scarcity_sensitivity

        if demand_regimes is None:
            demand_regimes = {"peak": 1.5, "normal": 1.0, "off_peak": 0.6}
        self.demand_regimes = demand_regimes
        self.regime_change_prob = regime_change_prob

        self.market_noise_scale = market_noise_scale
        self.burst_probability = burst_probability
        self.burst_min_size = burst_min_size
        self.burst_max_size = burst_max_size
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(self.prices))

        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([float(max_inventory), float(max_days)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._obs_buffer = np.empty(2, dtype=np.float32)
        self.inventory = 0
        self.days_left = 0
        self.total_revenue = 0.0
        self._burst = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days
        self.total_revenue = 0.0
        self._burst = 0
        self.current_regime = self.np_random.choice(list(self.demand_regimes.keys()))

        self._obs_buffer[0] = self.inventory
        self._obs_buffer[1] = self.days_left
        return self._obs_buffer, {}

    def get_demand(self, price: float) -> int:
        days_passed = self.max_days - self.days_left

        # 1. Regime multiplier
        regime_mult = self.demand_regimes[self.current_regime]

        # 2. S-curve urgency (sigmoid over normalized time)
        t = days_passed / self.max_days
        urgency_factor = 1.0 + self.urgency_amplitude / (
            1.0 + np.exp(-self.urgency_steepness * (t - self.urgency_midpoint))
        )

        # 3. Scarcity factor (FOMO)
        inventory_ratio = self.inventory / self.max_inventory
        scarcity_factor = 1.0 + self.scarcity_sensitivity * (1.0 - inventory_ratio)

        # 4. Per-segment demand
        total_expected = 0.0
        for segment in self.customer_segments:
            segment_expected = (
                self.base_demand
                * segment["weight"]
                * regime_mult
                * np.exp(segment["price_sensitivity"] * price)
                * urgency_factor
                * scarcity_factor
            )
            total_expected += segment_expected

        # 5. Market noise (log-normal)
        if self.market_noise_scale > 0:
            noise = np.exp(self.np_random.normal(0, self.market_noise_scale))
            total_expected *= noise

        # 6. Booking bursts
        self._burst = 0
        if self.np_random.random() < self.burst_probability:
            self._burst = int(self.np_random.integers(self.burst_min_size, self.burst_max_size + 1))
            total_expected += self._burst

        return self.np_random.poisson(total_expected)

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

        # Regime transition
        if not terminated and self.np_random.random() < self.regime_change_prob:
            keys = list(self.demand_regimes.keys())
            if len(keys) > 1:
                others = [k for k in keys if k != self.current_regime]
                self.current_regime = self.np_random.choice(others)

        self._obs_buffer[0] = self.inventory
        self._obs_buffer[1] = self.days_left

        info = {
            "total_revenue": self.total_revenue,
            "sold": sold,
            "demand": raw_demand,
            "price": price,
            "regime": self.current_regime,
            "burst": self._burst,
        }

        if self.render_mode == "human":
            self.render()

        return self._obs_buffer, revenue, terminated, False, info

    def render(self):
        message = (
            f"Day {self.max_days - self.days_left + 1:2d}/{self.max_days:2d} | "
            f"Inventory: {self.inventory:2d}/{self.max_inventory:2d} | "
            f"Regime: {self.current_regime:>8s} | "
            f"Revenue: {self.total_revenue:.2f}"
        )
        if self.render_mode == "human":
            print(message)
        elif self.render_mode == "ansi":
            return message

    def close(self) -> None:
        self.inventory = 0
        self.days_left = 0
        self.total_revenue = 0.0
