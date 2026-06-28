# 🔗 Dashboard Integration Notes — Member 2 Agents

> **Author:** Member 2
> **Date:** June 28, 2026
> **Purpose:** Guide for Member 4 to integrate Member 2's agents
> into the Streamlit dashboard

---

## How to Import Member 2's Agents

```python
from agents.time_based_agent import TimedBasedAgent
from agents.qlearning_agent  import QLearningAgent
from environment.airline_pricing_env import AirlinePricingEnv
```

---

## Available Data for Dashboard

### 1. Revenue Comparison Chart
```python
env      = AirlinePricingEnv()

# Time-Based Agent
tb_agent   = TimedBasedAgent(max_days=30, high_idx=15, low_idx=2)
tb_results = tb_agent.evaluate(env, num_episodes=100)
# Returns: {"mean_revenue": 296545, "std_revenue": 28456,
#           "min_revenue": 225000, "max_revenue": 358000}

# Q-Learning Agent
ql_agent   = QLearningAgent()
ql_agent.train(env, num_episodes=20000)
ql_results = ql_agent.evaluate(env, num_episodes=100)
# Returns: {"mean_revenue": 450000, "std_revenue": 52000,
#           "min_revenue": 25000,  "max_revenue": 479000}
```

### 2. Learning Curve Data
```python
# After training, episode rewards are stored here:
rewards = ql_agent.episode_rewards   # list of 20000 values
# Use this to plot learning curve in dashboard
```

### 3. Price Trajectory (Time-Based)
```python
prices = list(range(500, 10001, 500))  # price menu
days   = list(range(30, -1, -1))       # 30 to 0

trajectory = []
for d in days:
    action = tb_agent.select_action(d)
    trajectory.append(prices[action])
# Plot days vs trajectory in dashboard
```

### 4. Policy Heatmap (Q-Learning)
```python
import numpy as np
# After training:
policy = np.argmax(ql_agent.q_table, axis=2)
# Shape: (10, 10) — use as heatmap data
```

---

## Suggested Dashboard Sections

| Section | Data Source | Chart Type |
|---------|------------|------------|
| Revenue Comparison | `tb_results` vs `ql_results` | Bar chart |
| Learning Curve | `ql_agent.episode_rewards` | Line chart |
| Price Trajectory | `tb_agent.select_action()` | Line chart |
| Policy Heatmap | `ql_agent.q_table` | Heatmap |
| Improvement % | +46.60% | Metric card |

---

## Pre-generated Plot Files

These PNG files are already available in the repo root:

| File | Description |
|------|-------------|
| `qlearning_learning_curve.png` | Learning curve over 20,000 episodes |
| `qlearning_policy_heatmap.png` | Learned pricing policy heatmap |
| `time_based_trajectory.png` | Price decay over 30 days |

Member 4 can directly embed these in the dashboard if needed.

---

## Key Results to Display

| Metric | Value |
|--------|-------|
| Time-Based Mean Revenue | Rs 2,96,545 |
| Q-Learning Mean Revenue | Rs 4,50,000 |
| Improvement | **+46.60%** |
| Training Episodes | 20,000 |
| Convergence Episode | ~10,000 |

---

*Integration support: Contact Member 2 (Swarup) for any issues*
