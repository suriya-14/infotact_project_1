# =============================================================================
# time_based_agent.py
# Author      : Member 2
# Description : A rule-based heuristic baseline agent that adjusts price
#               based on how many days are left until departure.
#               This agent does NOT learn — it follows a fixed discount rule.
#               Purpose: Act as a performance baseline for Q-Learning & DQN.
# =============================================================================

# -----------------------------------------------------------------------------
# HOW THIS AGENT WORKS (Logic Plan)
# -----------------------------------------------------------------------------
# - At 30 days left  → charge MAX price (e.g., $300) — early bookers pay full
# - At 15 days left  → charge MID price (e.g., $200) — moderate discount
# - At  5 days left  → charge LOW price (e.g., $130) — urgency discount
# - At  1 day  left  → charge MIN price (e.g., $100) — clear remaining seats
#
# Formula:
#   price = max_price - ((max_price - min_price) / max_days) * days_elapsed
#
# This is a LINEAR decay from max_price → min_price over the booking window.
# -----------------------------------------------------------------------------

# TODO: Uncomment this import once Member 1 completes airline_pricing_env.py
# from environment.airline_pricing_env import AirlinePricingEnv

import numpy as np


class TimedBasedAgent:
    """
    A heuristic pricing agent that linearly decreases price over time.

    The core idea: the closer to departure, the lower the price —
    to ensure remaining inventory is cleared before the flight departs.

    Attributes:
        max_price  (float): Highest price charged (far from departure)
        min_price  (float): Lowest price charged (day of / day before departure)
        max_days   (int)  : Total days in the booking window (e.g., 30)
    """

    def __init__(self, max_price=300, min_price=100, max_days=30):
        """
        Initialize the Time-Based Agent with pricing bounds.

        Args:
            max_price (float): Starting price at the beginning of the window.
            min_price (float): Floor price as departure approaches.
            max_days  (int)  : Total length of the booking window in days.
        """
        self.max_price = max_price
        self.min_price = min_price
        self.max_days  = max_days

        # TODO: Add additional pricing tiers if needed
        # e.g., surge pricing on weekends, holiday multipliers

    def select_action(self, days_remaining):
        """
        Select a price based purely on how many days are left.

        Formula:
            price = max_price - ((max_price - min_price) / max_days)
                                * days_elapsed

        Args:
            days_remaining (int): How many days left until departure.

        Returns:
            float: The price to charge today.
        """
        # TODO: Implement the linear decay formula here
        # days_elapsed = self.max_days - days_remaining
        # price = self.max_price - (price_range / self.max_days) * days_elapsed
        # return round(price, 2)
        pass

    def evaluate(self, env, num_episodes=1000):
        """
        Run the agent in the environment for multiple episodes and
        collect total revenue statistics.

        Args:
            env         : The AirlinePricingEnv Gym environment (Member 1)
            num_episodes: Number of booking seasons to simulate

        Returns:
            dict: Summary stats — mean, std, min, max revenue
        """
        # TODO: Implement evaluation loop once env is available
        # revenues = []
        # for episode in range(num_episodes):
        #     state = env.reset()
        #     done  = False
        #     total_revenue = 0
        #
        #     while not done:
        #         days_remaining = state[1]           # index 1 = days_until_departure
        #         price          = self.select_action(days_remaining)
        #         state, reward, done, info = env.step(price)
        #         total_revenue += reward
        #
        #     revenues.append(total_revenue)
        #
        # return {
        #     "mean_revenue" : np.mean(revenues),
        #     "std_revenue"  : np.std(revenues),
        #     "min_revenue"  : np.min(revenues),
        #     "max_revenue"  : np.max(revenues),
        # }
        pass

    def plot_price_trajectory(self):
        """
        Plot how the price changes day-by-day over a 30-day booking window.
        Useful for visualizing the linear decay behavior.

        TODO: Implement using matplotlib once select_action() is done.
        """
        # import matplotlib.pyplot as plt
        # days  = list(range(self.max_days, 0, -1))
        # prices = [self.select_action(d) for d in days]
        # plt.plot(days, prices)
        # plt.xlabel("Days Remaining")
        # plt.ylabel("Price ($)")
        # plt.title("Time-Based Agent: Price Trajectory")
        # plt.gca().invert_xaxis()
        # plt.grid(True)
        # plt.show()
        pass


# -----------------------------------------------------------------------------
# QUICK TEST (run this file directly to test the agent standalone)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = TimedBasedAgent(max_price=300, min_price=100, max_days=30)

    print("Time-Based Agent initialized.")
    print(f"  Max Price : ${agent.max_price}")
    print(f"  Min Price : ${agent.min_price}")
    print(f"  Window    : {agent.max_days} days")
    print()
    print("Sample price decisions (days_remaining → price):")
    print("  [Will work after select_action() is implemented]")

    # TODO: Uncomment after implementing select_action()
    # for days in [30, 25, 20, 15, 10, 5, 1]:
    #     price = agent.select_action(days)
    #     print(f"  {days:2d} days left → ${price}")
