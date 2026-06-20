# =============================================================================
# time_based_agent.py
# Author      : Member 2
# Description : Rule-based heuristic baseline agent.
#               Price decays linearly as departure approaches.
#               Does NOT learn — acts as a baseline for Q-Learning & DQN.
# =============================================================================

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.airline_pricing_env import AirlinePricingEnv


class TimedBasedAgent:
    """
    A heuristic pricing agent that linearly decreases price over time.

    The closer to departure → the lower the price index chosen,
    to ensure remaining inventory clears before the flight departs.

    Attributes:
        prices        (list): Full price menu from the environment
        max_days      (int) : Total booking window in days (30)
        high_idx      (int) : Price index to start with (far from departure)
        low_idx       (int) : Price index to end with (near departure)
    """

    def __init__(self, max_days=30, high_idx=15, low_idx=2):
        """
        Args:
            max_days (int): Total days in the booking window
            high_idx (int): Starting price index (e.g., 15 = Rs 8000)
            low_idx  (int): Ending price index   (e.g.,  2 = Rs 1500)
        """
        # Full price menu: [500, 1000, 1500, ..., 10000]
        self.prices   = list(range(500, 10001, 500))
        self.max_days = max_days
        self.high_idx = high_idx   # index used far from departure
        self.low_idx  = low_idx    # index used near departure

    def select_action(self, days_remaining):
        """
        Select a price INDEX based on how many days are left.

        Linearly maps days_remaining → price index:
            days_remaining = 30  →  high_idx (expensive)
            days_remaining =  0  →  low_idx  (cheap)

        Args:
            days_remaining (int): Days left until departure

        Returns:
            int: Price index into the environment's price list
        """
        days_remaining = max(0, min(days_remaining, self.max_days))

        # Linear interpolation from high_idx to low_idx
        ratio      = days_remaining / self.max_days
        action_idx = int(self.low_idx + ratio * (self.high_idx - self.low_idx))
        action_idx = max(0, min(action_idx, len(self.prices) - 1))

        return action_idx

    def evaluate(self, env, num_episodes=1000):
        """
        Run agent across multiple episodes and collect revenue stats.

        Args:
            env          : AirlinePricingEnv instance
            num_episodes : Number of booking seasons to simulate

        Returns:
            dict: mean, std, min, max revenue
        """
        revenues = []

        for episode in range(num_episodes):
            obs, _        = env.reset()
            done          = False
            total_revenue = 0.0

            while not done:
                days_remaining          = int(obs[1])
                action                  = self.select_action(days_remaining)
                obs, reward, terminated, truncated, info = env.step(action)
                done                    = terminated or truncated
                total_revenue          += reward

            revenues.append(total_revenue)

            if (episode + 1) % 100 == 0:
                print(f"  Episode {episode+1}/{num_episodes} | "
                      f"Avg Revenue: Rs {np.mean(revenues):.2f}")

        return {
            "mean_revenue" : round(np.mean(revenues), 2),
            "std_revenue"  : round(np.std(revenues), 2),
            "min_revenue"  : round(np.min(revenues), 2),
            "max_revenue"  : round(np.max(revenues), 2),
        }

    def plot_price_trajectory(self):
        """Plot how price changes day-by-day over the booking window."""
        import matplotlib.pyplot as plt

        days   = list(range(self.max_days, -1, -1))
        prices = [self.prices[self.select_action(d)] for d in days]

        plt.figure(figsize=(10, 5))
        plt.plot(days, prices, marker='o', color='blue', label='Time-Based Price')
        plt.xlabel("Days Remaining Until Departure")
        plt.ylabel("Price (Rs)")
        plt.title("Time-Based Agent: Price Trajectory")
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("time_based_trajectory.png")
        plt.show()
        print("Plot saved as time_based_trajectory.png")


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Time-Based Agent — Test Run")
    print("=" * 50)

    agent = TimedBasedAgent(max_days=30, high_idx=15, low_idx=2)

    print("\nPrice decisions by day:")
    print(f"{'Days Left':<12} {'Action Idx':<12} {'Price (Rs)':<12}")
    print("-" * 36)
    for days in [30, 25, 20, 15, 10, 5, 1, 0]:
        idx   = agent.select_action(days)
        price = agent.prices[idx]
        print(f"{days:<12} {idx:<12} {price:<12}")

    print("\nRunning 100-episode evaluation...")
    env     = AirlinePricingEnv()
    results = agent.evaluate(env, num_episodes=100)

    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: Rs {v}")
