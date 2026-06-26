# ✈️ RL Dynamic Pricing — Airline Ticket Pricing with Reinforcement Learning

> A multi-agent reinforcement learning project that simulates dynamic airline ticket pricing strategies using a custom Gymnasium environment.

---

## 👥 Team

| Member | Role | Contribution |
|--------|------|--------------|
| Member 1 | Environment Architect | Custom Gym environment + Fixed Price baseline agent |
| Member 2 | Classical Agents | Time-based pricing agent + Q-Learning agent |
| Member 3 | Deep RL Agents | Inventory-based heuristic agent + DQN agent |
| Member 4 | Dashboard | Streamlit simulation dashboard |

---

## 📁 Project Structure

```
rl-dynamic-pricing/
│
├── environment/
│   └── airline_pricing_env.py       # Custom Gymnasium environment (Member 1)
│
├── agents/
│   ├── fixed_price_agent.py         # Baseline fixed price agent (Member 1)
│   ├── time_based_agent.py          # Time-based pricing agent (Member 2)
│   ├── qlearning_agent.py           # Tabular Q-Learning agent (Member 2)
│   ├── inventory_based_agent.py     # Inventory heuristic agent (Member 3)
│   └── dqn_agent.py                 # Deep Q-Network (DQN) agent (Member 3)
│
├── models/
│   └── dqn_weights.pth              # Trained DQN model weights (Member 3)
│
├── dashboard/
│   └── app.py                       # Streamlit simulation dashboard (Member 4)
│
└── README.md
```

---

## ⚙️ Environment — `airline_pricing_env.py` (Member 1)

A custom **Gymnasium** environment simulating airline ticket demand over a booking window.

| Parameter | Value |
|-----------|-------|
| `state_dim` | 2 (days remaining, inventory remaining) |
| `action_dim` | 20 |
| Price Range | ₹500 – ₹10,000 |

**State:** `[days_remaining, inventory_remaining]`  
**Action:** Index into a discrete price ladder (20 price levels)  
**Reward:** Revenue collected per step based on demand response to price  

---

## 🤖 Agents

### Member 1 — Fixed Price Agent (`fixed_price_agent.py`)
A simple baseline that applies a constant price throughout the booking window. Used as a performance benchmark for all other agents.

---

### Member 2 — Time-Based Agent (`time_based_agent.py`)
Adjusts pricing based on how close the flight date is. Applies higher prices far from departure and discounts as the date approaches to clear remaining seats.

### Member 2 — Q-Learning Agent (`qlearning_agent.py`)
A tabular reinforcement learning agent that learns a Q-table mapping discretized states to optimal price actions through trial-and-error interaction with the environment.

---

### Member 3 — Inventory-Based Agent (`inventory_based_agent.py`)
A rule-based heuristic agent that reacts to inventory levels in real time:
- **Scarcity mode:** Raises prices when inventory is low
- **Clearance mode:** Lowers prices when time is running out and seats remain

**Constructor parameters:**
```python
InventoryBasedAgent(
    prices,           # list of 20 price levels
    max_inventory,    # total seats available
    max_days,         # total booking window
    scarcity_weight,  # how aggressively to raise price under scarcity
    urgency_weight    # how aggressively to drop price near departure
)
```

**Key methods:** `act()`, `get_price()`, `update()`, `reset()`

---

### Member 3 — DQN Agent (`dqn_agent.py`)
A neural network-based deep RL agent trained with **PyTorch** using experience replay and a target network.

**Architecture:**

| Class | Role |
|-------|------|
| `ReplayBuffer` | Stores past transitions for experience replay |
| `QNetwork` | Neural net mapping state → Q-values for all 20 actions |
| `DQNAgent` | Full training and inference logic |

**Inference config:** `epsilon = 0` (greedy policy, no exploration)  
**Trained weights:** `models/dqn_weights.pth`

**Loading for inference:**
```python
from agents.dqn_agent import DQNAgent

agent = DQNAgent(state_dim=2, action_dim=20)
agent.load("models/dqn_weights.pth")
action = agent.act(state)
```

---

### Member 4 — Streamlit Dashboard (`dashboard/app.py`)

An interactive web dashboard to simulate and compare all agents visually.

**Features:**
- Agent selector (choose which agent to simulate)
- Day-by-day simulation table
- Price trajectory chart
- Revenue comparison chart across all agents
- 1000-simulation Monte Carlo mode for statistical comparison
- KPI cards (total revenue, avg price, seats sold)
- Optional inventory heatmap

**Run the dashboard:**
```bash
streamlit run dashboard/app.py
```

**Handoff parameters from Member 3:**
```python
state_dim   = 2
action_dim  = 20
price_min   = 500    # ₹
price_max   = 10000  # ₹
weights_path = "models/dqn_weights.pth"
```

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-repo>/rl-dynamic-pricing.git
cd rl-dynamic-pricing

# Install dependencies
pip install gymnasium torch numpy streamlit

# Train the DQN agent (optional — weights already included)
python agents/dqn_agent.py

# Launch the dashboard
streamlit run dashboard/app.py
```

---

## 📊 Agent Comparison (Expected)

| Agent | Strategy | Strengths |
|-------|----------|-----------|
| Fixed Price | Constant pricing | Simple baseline |
| Time-Based | Date-proximity discounting | Predictable, interpretable |
| Q-Learning | Tabular RL | Learns from experience |
| Inventory-Based | Stock-aware heuristic | Real-time reactive |
| DQN | Deep RL | Best generalization |

---

## 🏫 Project Info

**Course:** B.Tech — Artificial Intelligence & Data Science  
**Institution:** SNS College of Engineering, Coimbatore  
**Type:** Internship Project — Reinforcement Learning  

---
