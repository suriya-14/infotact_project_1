# ✈️ Airline Dynamic Pricing using Reinforcement Learning

A multi-agent reinforcement learning system that simulates and compares intelligent pricing strategies for airline ticket sales. Built as a team project with each member contributing a distinct component.

---

## 📁 Project Structure

```
airline-rl-pricing/
├── environment/
│   └── airline_pricing_env.py       # Custom OpenAI Gymnasium environment (Member 1)
├── agents/
│   ├── fixed_price_agent.py         # Baseline fixed price agent (Member 1)
│   ├── time_based_agent.py          # Time-decay pricing agent (Member 2)
│   ├── qlearning_agent.py           # Tabular Q-Learning agent (Member 2)
│   ├── inventory_based_agent.py     # Heuristic inventory agent (Member 3)
│   └── dqn_agent.py                 # Deep Q-Network agent (Member 3)
├── models/
│   └── dqn_weights.pth              # Trained DQN model weights (Member 3)
├── dashboard/
│   └── app.py                       # Streamlit comparison dashboard (Member 4)
└── README.md
```

---

## 🧠 Agent Overview

| Agent | Type | Member | Description |
|---|---|---|---|
| `fixed_price_agent.py` | Baseline | M1 | Sets a constant price throughout the episode |
| `time_based_agent.py` | Heuristic | M2 | Adjusts price based on days remaining |
| `qlearning_agent.py` | RL (Tabular) | M2 | Learns a Q-table via Bellman updates |
| `inventory_based_agent.py` | Heuristic | M3 | Prices dynamically using scarcity + urgency logic |
| `dqn_agent.py` | RL (Deep) | M3 | Trains a neural Q-network with experience replay |

---

## 🔧 Environment

The custom `AirlinePricingEnv` (Gymnasium-compatible) simulates a flight booking window.

- **State space:** `[inventory_left, days_left]` — shape `(2,)`
- **Action space:** 20 discrete price levels — `₹500` to `₹10,000` in steps of `₹500`
- **Reward:** Revenue earned per timestep based on demand response to price

---

## 📦 Member 3 — Inventory-Based & DQN Agents

### InventoryBasedAgent (`inventory_based_agent.py`)

A heuristic agent that sets prices based on remaining inventory and days left — no learning involved. Used as a smart baseline in the dashboard comparison.

**Pricing Logic:**
- **Low inventory (scarcity)** → push price up
- **High inventory + few days left (urgency)** → push price down to clear stock
- **Otherwise** → stay near a mid-range price

**Key Parameters:**
```python
InventoryBasedAgent(
    prices=list(range(500, 10001, 500)),
    max_inventory=100,
    max_days=30,
    scarcity_weight=0.6,
    urgency_weight=0.4
)
```

**Methods:**

| Method | Description |
|---|---|
| `act(state)` | Returns action index (price level) |
| `get_price(state)` | Returns actual price in ₹ |
| `update(...)` | No-op (heuristic, no learning) |
| `reset()` | No-op |

---

### DQNAgent (`dqn_agent.py`)

A Deep Q-Network agent trained end-to-end on the airline pricing environment using experience replay and a target network.

**Architecture:**

```
Input (2) → Linear(64) → ReLU → Linear(64) → ReLU → Output (20)
```

**Key Components:**
- `ReplayBuffer` — stores transitions, samples random mini-batches
- `QNetwork` — 2-layer MLP predicting Q-values for all 20 actions
- `DQNAgent` — manages training loop, epsilon-greedy exploration, target network sync

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `state_dim` | 2 |
| `action_dim` | 20 |
| `learning_rate` | 0.001 |
| `gamma` | 0.99 |
| `epsilon_decay` | 0.995 |
| `batch_size` | 64 |
| `buffer_capacity` | 10,000 |
| `target_update` | Every 10 episodes |
| `training_episodes` | 1,000 |

**Training:**
```bash
python agents/dqn_agent.py
# Weights saved to models/dqn_weights.pth
```

**Inference (epsilon=0, greedy):**
```python
agent = DQNAgent(state_dim=2, action_dim=20, epsilon=0)
agent.load("models/dqn_weights.pth")
action = agent.select_action(state, epsilon=0)
```

---

## 🔗 Member 4 Dashboard Integration

Parameters to pass when loading the DQN agent in `dashboard/app.py`:

```python
state_dim    = 2
action_dim   = 20
prices       = list(range(500, 10001, 500))
epsilon      = 0        # greedy inference, no exploration
weights_path = "models/dqn_weights.pth"
```

The dashboard is expected to support:
- Day-by-day simulation table
- Agent selector (all 5 agents)
- Price trajectory chart
- Revenue comparison chart
- 1,000-simulation Monte Carlo mode
- KPI cards (total revenue, avg price, occupancy)

---

## ⚙️ Setup & Installation

```bash
pip install torch numpy gymnasium
```

To run the DQN training from scratch:
```bash
python agents/dqn_agent.py
```

Pre-trained weights are already available at `models/dqn_weights.pth`.

---

## 👥 Team

| Member | Component |
|---|---|
| Member 1 | Gymnasium Environment + Fixed Price Agent |
| Member 2 | Time-Based Agent + Q-Learning Agent |
| **Member 3** | **Inventory-Based Agent + DQN Agent + Trained Weights** |
| Member 4 | Streamlit Dashboard |

---

## 📌 Notes

- All agents implement a common interface: `act(state)`, `update(...)`, `reset()`
- The DQN agent uses `map_location` in `torch.load` for CPU/GPU portability
- Prices are indexed as actions: `action 0 = ₹500`, `action 19 = ₹10,000`
