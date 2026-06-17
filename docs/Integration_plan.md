# 🔗 Integration Plan — Connecting with Member 1's Environment

> **Author:** Member 2  
> **Date:** June 17, 2026  
> **Status:** Ready to integrate — Member 1's `airline_pricing_env.py` is complete  

---

## Overview

Member 1 has completed `airline_pricing_env.py` in their branch. This document
outlines the integration steps I will follow tomorrow to connect my agents
(`time_based_agent.py` and `qlearning_agent.py`) with the shared environment
and begin live training runs.

---

## Step 1 — Pull Member 1's Environment

```bash
# Fetch all branches
git fetch origin

# Merge Member 1's branch into my Swarup branch
git merge origin/<member1-branch-name>
```

Or on GitHub:
- Go to my `Swarup` branch
- Click **"Compare & pull request"**
- Merge Member 1's branch changes in

---

## Step 2 — Verify Environment Interface

Before running my agents, I will confirm the environment exposes the standard
Gym interface my agents expect:

```python
from environment.airline_pricing_env import AirlinePricingEnv

env = AirlinePricingEnv()

# Check reset() returns (inventory, days_remaining)
state = env.reset()
print("Initial state:", state)   # Expected: (100, 30) or similar

# Check step() returns (next_state, reward, done, info)
next_state, reward, done, info = env.step(200)  # price = $200
print("Next state :", next_state)
print("Reward     :", reward)
print("Done       :", done)
```

---

## Step 3 — Uncomment Environment Imports in My Agents

In `time_based_agent.py`:
```python
# UNCOMMENT this line:
from environment.airline_pricing_env import AirlinePricingEnv
```

In `qlearning_agent.py`:
```python
# UNCOMMENT this line:
from environment.airline_pricing_env import AirlinePricingEnv
```

---

## Step 4 — Run Time-Based Agent First (Simpler)

Since `time_based_agent.py` has no training loop, it's the fastest to test:

```python
from environment.airline_pricing_env import AirlinePricingEnv
from agents.time_based_agent import TimedBasedAgent

env   = AirlinePricingEnv()
agent = TimedBasedAgent(max_price=300, min_price=100, max_days=30)

# Single episode test
state         = env.reset()
done          = False
total_revenue = 0

while not done:
    days_remaining          = state[1]
    price                   = agent.select_action(days_remaining)
    state, reward, done, _  = env.step(price)
    total_revenue          += reward

print(f"Total Revenue (1 episode): ${total_revenue:.2f}")
```

**Expected outcome:** A single revenue number (e.g., $7,800–$9,500 range)

---

## Step 5 — Run Q-Learning Training Loop

Once the environment is confirmed working, implement and run `train()`:

```python
from environment.airline_pricing_env import AirlinePricingEnv
from agents.qlearning_agent import QLearningAgent

env   = AirlinePricingEnv()
agent = QLearningAgent(
    price_levels  = [100, 150, 200, 250, 300],
    max_inventory = 100,
    max_days      = 30,
    alpha         = 0.1,
    gamma         = 0.95,
    epsilon       = 1.0,
    epsilon_decay = 0.995,
)

# Train for 10,000 episodes
rewards = agent.train(env, num_episodes=10000)
print(f"Final avg reward (last 100 eps): ${sum(rewards[-100:])/100:.2f}")
```

---

## Step 6 — Evaluate and Compare

After training, run evaluation against the baseline:

```python
# Evaluate Q-Learning agent
ql_results = agent.evaluate(env, num_episodes=1000)
print("Q-Learning Results:", ql_results)

# Evaluate Time-Based baseline
tb_results = time_based_agent.evaluate(env, num_episodes=1000)
print("Time-Based Results:", tb_results)
```

**Target:** Q-Learning mean revenue > Time-Based mean revenue ✅

---

## Checklist for Tomorrow

- [ ] Pull Member 1's branch / merge environment
- [ ] Verify `env.reset()` and `env.step()` work correctly
- [ ] Uncomment environment imports in both agent files
- [ ] Run single-episode test with `time_based_agent.py`
- [ ] Implement `train()` in `qlearning_agent.py`
- [ ] Run 10,000 episode training
- [ ] Implement `evaluate()` and compare both agents
- [ ] Commit results

---

## Questions to Clarify with Member 1

- What is the exact state format returned by `env.reset()`? → `(inventory, days)` tuple or numpy array?
- What is the action format expected by `env.step(action)`? → raw price float or price index?
- What is `max_inventory` and `max_days` set to in the environment?
- Is reward returned as `price * units_sold` or something different?

---

*Integration begins: June 18, 2026*
