# =============================================================================
# run.py
# Author      : Member 2
# Description : Main script to train Q-Learning agent and compare
#               performance against Time-Based baseline agent.
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
    # 1. Evaluate Time-Based Baseline (no training needed)
    # ------------------------------------------------------------------
    print("\n[1] Running Time-Based Baseline Agent...")
    tb_agent  = TimedBasedAgent(max_days=30, high_idx=15, low_idx=2)
    tb_results = tb_agent.evaluate(env, num_episodes=1000)

    # ------------------------------------------------------------------
    # 2. Train Q-Learning Agent
    # ------------------------------------------------------------------
    print("\n[2] Training Q-Learning Agent...")
    ql_agent  = QLearningAgent(
        max_inventory = 50,
        max_days      = 30,
        num_actions   = 20,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
    )
    ql_agent.train(env, num_episodes=10000)

    # ------------------------------------------------------------------
    # 3. Evaluate Q-Learning Agent
    # ------------------------------------------------------------------
    print("\n[3] Evaluating Q-Learning Agent...")
    ql_results = ql_agent.evaluate(env, num_episodes=1000)

    # ------------------------------------------------------------------
    # 4. Compare Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  RESULTS COMPARISON (1000 episodes)")
    print("=" * 55)
    print(f"{'Metric':<20} {'Time-Based':>15} {'Q-Learning':>15}")
    print("-" * 55)
    print(f"{'Mean Revenue':<20} Rs {tb_results['mean_revenue']:>12,.2f} Rs {ql_results['mean_revenue']:>12,.2f}")
    print(f"{'Std Revenue':<20} Rs {tb_results['std_revenue']:>12,.2f} Rs {ql_results['std_revenue']:>12,.2f}")
    print(f"{'Min Revenue':<20} Rs {tb_results['min_revenue']:>12,.2f} Rs {ql_results['min_revenue']:>12,.2f}")
    print(f"{'Max Revenue':<20} Rs {tb_results['max_revenue']:>12,.2f} Rs {ql_results['max_revenue']:>12,.2f}")
    print("=" * 55)

    improvement = ((ql_results['mean_revenue'] - tb_results['mean_revenue'])
                   / tb_results['mean_revenue'] * 100)
    print(f"\nQ-Learning improvement over baseline: {improvement:.2f}%")

    if improvement > 0:
        print("Q-Learning OUTPERFORMS the Time-Based baseline ✅")
    else:
        print("Needs more training — try increasing num_episodes ⚠️")

    # ------------------------------------------------------------------
    # 5. Plot Results
    # ------------------------------------------------------------------
    print("\n[4] Generating plots...")
    ql_agent.plot_learning_curve()
    ql_agent.plot_policy()
    tb_agent.plot_price_trajectory()

    print("\nDone! Check the generated .png files.")


if __name__ == "__main__":
    main()
