# ✈️ Dynamic Pricing with Reinforcement Learning — Member 2 Contributions

> **Project:** Travel & Hospitality — RL-Based Dynamic Pricing Agent  
> **Role:** Member 2 — Baseline Heuristic Agent (Time-Based) + Q-Learning RL Agent  
> **Team Project:** Airline/Hotel inventory pricing using OpenAI Gym + Deep Q-Networks

---

## 📌 Overview

This repository contains my individual contributions to a team reinforcement learning project that builds an autonomous pricing agent for airline/hotel seat inventory. The agent learns to dynamically adjust prices over a 30-day booking window to **maximize total revenue** while ensuring inventory clears before the departure date.

My role covers two core components:
- `agents/time_based_agent.py` — A rule-based heuristic baseline that discounts prices as departure approaches
- `agents/qlearning_agent.py` — A tabular Q-Learning agent that learns the optimal pricing policy through environment interaction

---

## 🗂️ Project Structure (Full Team)

```
project/
│
├── agents/
│   ├── fixed_price_agent.py         ← Member 1
│   ├── time_based_agent.py          ← Member 2 (Me) ✅
│   ├── inventory_based_agent.py     ← Member 3
│   ├── qlearning_agent.py           ← Member 2 (Me) ✅
│   └── dqn_agent.py                 ← Member 3
│
├── environment/
│   └── airline_pricing_env.py       ← Member 1
│
├── models/
│   └── dqn_weights.pth              ← Member 3
│
├── dashboard/
│   └── app.py                       ← Member 4 (Streamlit)
│
└── README.md
```

---

## 🧠 My Deliverables

### 1. `time_based_agent.py` — Heuristic Baseline

A **rule-based pricing strategy** that requires no learning. It linearly discounts the price as the departure date approaches — mimicking a common real-world airline pricing pattern.

**Logic:**
- Starts at a maximum price (e.g., $300) at 30 days out
- Reduces price proportionally as days remaining decrease
- Applies a steeper discount in the final days to clear remaining inventory

**Purpose:** Acts as a performance baseline. The Q-Learning and DQN agents must beat this to prove the value of RL.

---

### 2. `qlearning_agent.py` — Tabular Q-Learning Agent

A **reinforcement learning agent** that learns an optimal pricing policy by building a Q-table over the state-action space.

**MDP Formulation:**

| Component | Definition |
|-----------|-----------|
| **State** | `(remaining_inventory, days_until_departure)` |
| **Action** | Discrete price level (e.g., $100 to $300 in steps) |
| **Reward** | Revenue earned = `price × units_sold` at each timestep |
| **Policy** | `argmax Q(s, a)` after convergence |

**Key Implementation Details:**
- Q-table initialized to zeros; updated via the **Bellman equation**
- **Epsilon-greedy** exploration: starts at ε=1.0, decays to ε=0.01 over training
- Hyperparameters: learning rate `α = 0.1`, discount factor `γ = 0.95`
- Trained over **10,000 episodes** until convergence

**Bellman Update Rule:**
```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') − Q(s, a)]
```

---

## 📊 Performance Results

After running 1,000 evaluation episodes against all baseline agents:

| Agent | Avg Revenue / Season |
|-------|---------------------|
| Fixed Price Agent | ~$8,400 |
| **Time-Based Agent (Mine)** | **~$9,100** |
| **Q-Learning Agent (Mine)** | **~$11,200** |
| DQN Agent | ~$12,800 |

> ✅ Q-Learning outperforms both heuristic baselines, validating the RL approach.

---

## 🚀 How to Run My Code

### Prerequisites
```bash
pip install gymnasium numpy matplotlib
```

### Run Time-Based Agent
```bash
python agents/time_based_agent.py
```

### Train Q-Learning Agent
```bash
python agents/qlearning_agent.py --train --episodes 10000
```

### Evaluate Q-Learning Agent
```bash
python agents/qlearning_agent.py --evaluate --episodes 1000
```

---

## 📅 Weekly Progress Log

### Week 1 — Environment Understanding & Problem Formulation
- Studied the MDP formulation for the airline pricing problem
- Reviewed `airline_pricing_env.py` (Member 1's Gym environment)
- Identified state space: `[remaining_inventory, days_until_departure]`
- Mapped action space: discrete price levels from $100–$300

### Week 2 — Baseline Agent + Q-Learning Implementation
- ✅ Implemented `time_based_agent.py` with linear price decay logic
- ✅ Initialized Q-table with shape `(max_inventory, max_days, num_price_levels)`
- ✅ Implemented epsilon-greedy exploration with decay schedule
- ✅ Implemented Bellman equation update loop
- ✅ Ran initial training for 5,000 episodes; observed Q-table convergence

### Week 3 — Tuning, Debugging & Evaluation
- Tuned hyperparameters: `α`, `γ`, epsilon decay rate
- Added logging to track cumulative reward per episode
- Plotted learning curve: reward vs. episodes (using Matplotlib)
- Compared Q-Learning vs. Time-Based baseline across 1,000 eval episodes

### Week 4 — Integration & Documentation
- Integrated agents with team codebase and shared Gym environment
- Verified compatibility with Member 3's DQN agent and Member 4's dashboard
- Wrote unit tests for Q-table update logic
- Finalized README and pushed all code + results to GitHub

---

## 📈 Key Visualizations

### Learning Curve (Q-Learning Training)
Shows cumulative reward per episode increasing as the agent learns the optimal policy over 10,000 training episodes.

### Price Trajectory (Evaluation)
Demonstrates that the Q-Learning agent learns complex behaviors — holding prices high early in the booking window, then aggressively discounting in the final 3–5 days to clear remaining inventory.

> Plots are generated via `matplotlib` and saved to `/results/` folder.

---

## 🔑 Key Concepts Applied

- **Markov Decision Process (MDP):** Formal framework for sequential decision-making under uncertainty
- **Q-Learning:** Model-free, off-policy RL algorithm using a value table
- **Bellman Equation:** Recursive update rule for Q-value estimation
- **Epsilon-Greedy Exploration:** Balances exploration of new prices vs. exploitation of learned policy
- **Stochastic Demand Modeling:** Purchase probability decreases with price and varies with time
- **Revenue Management:** Balancing margin maximization vs. inventory spoilage prevention

---

## 🤝 Team Members

| Member | Role |
|--------|------|
| Member 1 | Gym Environment + Fixed Price Agent |
| **Member 2 (Me)** | **Time-Based Agent + Q-Learning Agent** |
| Member 3 | Inventory-Based Agent + DQN Agent |
| Member 4 | Streamlit Dashboard |

---

## 📚 References

- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd Edition)
- OpenAI Gymnasium Documentation — [gymnasium.farama.org](https://gymnasium.farama.org)
- Watkins & Dayan (1992) — *Q-learning*, Machine Learning Journal
- Talluri & van Ryzin — *The Theory and Practice of Revenue Management*

---

*Last updated: June 2026 | Part of an internship project on RL-based dynamic pricing in Travel & Hospitality*
