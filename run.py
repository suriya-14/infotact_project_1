# =============================================================================
# run.py
# Author      : Member 2
# Description : Train Q-Learning agent and compare against Time-Based baseline
# Run         : python run.py
# =============================================================================

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.airline_pricing_env import AirlinePricingEnv
from agents.time_based_agent import TimedBasedAgent
from agents.qlearning_agent  import QLearningAgent


def main():
    print("=" * 55)
    print("  RL Dynamic Pricing — Member 2 Training Run")
    print("=" * 55)

    env = AirlinePricingEnv()

    # ------------------------------------------------------------------
    # 1. Evaluate Time-Based Baseline
    # ------------------------------------------------------------------
    print("\n[1] Running Time-Based Baseline Agent...")
    tb_agent   = TimedBasedAgent(max_days=30, high_idx=15, low_idx=2)
    tb_results = tb_agent.evaluate(env, num_episodes=1000)

    # ------------------------------------------------------------------
    # 2. Train Q-Learning Agent
    # ------------------------------------------------------------------
    print("\n[2] Training Q-Learning Agent...")
    ql_agent = QLearningAgent(
        max_inventory       = 50,
        max_days            = 30,
        num_actions         = 20,
        n_inventory_buckets = 10,   # bucket state space for faster learning
        n_day_buckets       = 10,
        alpha               = 0.2,  # higher learning rate
        gamma               = 0.99, # value future rewards highly
        epsilon             = 1.0,
        epsilon_min         = 0.01,
        epsilon_decay       = 0.9995, # slower decay = more exploration
    )
    ql_agent.train(env, num_episodes=20000)

    # ------------------------------------------------------------------
    # 3. Evaluate Q-Learning Agent
    # ------------------------------------------------------------------
    print("\n[3] Evaluating Q-Learning Agent...")
    ql_results = ql_agent.evaluate(env, num_episodes=1000)

    # ------------------------------------------------------------------
    # 4. Compare Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  RESULTS COMPARISON (1000 evaluation episodes)")
    print("=" * 55)
    print(f"{'Metric':<20} {'Time-Based':>15} {'Q-Learning':>15}")
    print("-" * 55)
    for key in ["mean_revenue", "std_revenue", "min_revenue", "max_revenue"]:
        print(f"{key:<20} Rs {tb_results[key]:>12,.2f} Rs {ql_results[key]:>12,.2f}")
    print("=" * 55)

    improvement = ((ql_results['mean_revenue'] - tb_results['mean_revenue'])
                   / tb_results['mean_revenue'] * 100)
    print(f"\nQ-Learning improvement over baseline: {improvement:.2f}%")

    if improvement > 0:
        print("Q-Learning OUTPERFORMS the Time-Based baseline ✅")
    else:
        print("Still training needed — increase num_episodes ⚠️")

    # ------------------------------------------------------------------
    # 5. Plot Results
    # ------------------------------------------------------------------
    print("\n[4] Generating plots...")
    ql_agent.plot_learning_curve()
    ql_agent.plot_policy()
    tb_agent.plot_price_trajectory()

    print("\nDone! Check the .png files in your project folder.")


if __name__ == "__main__":
    main()
