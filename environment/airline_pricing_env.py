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

    def __init__(
        self,
        max_inventory: int = 50,
        max_days: int = 30,
        prices: list = None,
        base_demand: float = 10.0,
        price_sensitivity: float = -0.0003,
        urgency_factor_slope: float = 0.05,
        render_mode: str = None,
    ):
        """
        Initializes the Airline Pricing Environment.

        Args:
            max_inventory (int): Total initial inventory to sell.
            max_days (int): Time horizon (days/steps until departure).
            prices (list): Available pricing levels (actions).
            base_demand (float): Base demand scaling factor.
            price_sensitivity (float): Price elasticity coefficient (negative value).
            urgency_factor_slope (float): Linear scaling for urgency as deadline nears.
            render_mode (str): Rendering mode ("human", "ansi", or None).
        """
        super().__init__()
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.prices = prices if prices is not None else [2000, 3000, 4000, 5000, 6000]
        self.base_demand = base_demand
        self.price_sensitivity = price_sensitivity
        self.urgency_factor_slope = urgency_factor_slope
        self.render_mode = render_mode

        # Define action space: index of the price level chosen
        self.action_space = gym.spaces.Discrete(len(self.prices))

        # Define observation space: [remaining_inventory, days_until_departure]
        # Low is [0, 0] and High is [max_inventory, max_days]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([float(max_inventory), float(max_days)], dtype=np.float32),
            dtype=np.float32,
        )

        # Environment variables initialized in reset
        self.inventory = 0
        self.days_left = 0
        self.total_revenue = 0.0

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets the environment to an initial state.

        Args:
            seed (int): Random seed for reproducibility.
            options (dict): Additional options for reset (unused).

        Returns:
            observation (np.ndarray): Initial state [max_inventory, max_days].
            info (dict): Auxiliary information.
        """
        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days
        self.total_revenue = 0.0

        observation = np.array([self.inventory, self.days_left], dtype=np.float32)
        return observation, {}

    def get_demand(self, price: float) -> int:
        """
        Calculates customer demand (bookings) at a given price using a
        stochastic Poisson process influenced by price and days remaining.

        Args:
            price (float): Chosen ticket/room price.

        Returns:
            int: Stochastic bookings count.
        """
        # Urgency rises as days left decreases (30 - days_left)
        days_passed = self.max_days - self.days_left
        urgency_factor = 1.0 + (self.urgency_factor_slope * days_passed)
        
        # Expected bookings rate using exponential price sensitivity
        expected_bookings = (
            self.base_demand
            * np.exp(self.price_sensitivity * price)
            * urgency_factor
        )

        # Gymnasium's self.np_random generator ensures reproducibility when seeded
        return self.np_random.poisson(expected_bookings)

    def step(self, action: int):
        """
        Steps the environment by executing the pricing action.

        Args:
            action (int): Index of the pricing level.

        Returns:
            observation (np.ndarray): Next state [inventory, days_left].
            reward (float): Revenue earned in this step.
            terminated (bool): Whether the episode has terminated (out of inventory or time).
            truncated (bool): Whether the episode was truncated (always False here).
            info (dict): Auxiliary information including total revenue.
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in {self.action_space}")

        price = self.prices[action]
        raw_demand = self.get_demand(price)
        
        # Bookings cannot exceed remaining inventory
        sold = min(raw_demand, self.inventory)
        revenue = float(price * sold)

        self.inventory -= sold
        self.days_left -= 1
        self.total_revenue += revenue

        terminated = (self.inventory <= 0) or (self.days_left <= 0)
        truncated = False

        observation = np.array([self.inventory, self.days_left], dtype=np.float32)
        info = {
            "total_revenue": self.total_revenue,
            "sold": sold,
            "demand": raw_demand,
            "price": price,
        }

        if self.render_mode == "human":
            self.render()

        return observation, revenue, terminated, truncated, info

    def render(self):
        """
        Renders the current state of the environment.
        """
        message = (
            f"Days Left: {self.days_left:2d} | "
            f"Inventory: {self.inventory:2d}/{self.max_inventory:2d} | "
            f"Episodic Revenue: {self.total_revenue:.2f}"
        )
        if self.render_mode == "human":
            print(message)
        elif self.render_mode == "ansi":
            return message
