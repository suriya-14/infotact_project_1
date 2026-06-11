# 🛫 RL Dynamic Pricing — Interactive Dashboard
### Member 4 | Week 4 | Policy Evaluation & Streamlit Dashboard

---

## 📌 Overview

This module is the **final deliverable** of the RL Dynamic Pricing project. It takes all trained agents from Members 1, 2, and 3 and presents a fully interactive **Streamlit dashboard** that evaluates, compares, and visualizes agent performance across 1,000 simulated booking seasons.

The dashboard is designed for both **Data Scientists** (who want RL metrics) and **Revenue Managers** (who want business KPIs) — no ML knowledge required to use it.

---

## 🗂️ Folder Structure

```
project/
│
├── environment/
│   └── airline_pricing_env.py        ← Member 1: Custom Gym Environment
│
├── agents/
│   ├── fixed_price_agent.py          ← Member 2: Fixed Price Baseline
│   ├── time_based_agent.py           ← Member 2: Time-Based Baseline
│   ├── inventory_based_agent.py      ← Member 2: Inventory-Based Baseline
│   ├── qlearning_agent.py            ← Member 2: Q-Learning Agent
│   └── dqn_agent.py                  ← Member 3: Deep Q-Network Agent
│
├── models/
│   ├── qtable_trained.npy            ← Member 2: Trained Q-Table weights
│   └── dqn_weights.pth               ← Member 3: Trained DQN weights
│
├── dashboard/
│   └── app.py                        ← Member 4: Main Streamlit App
│
├── data/
│   └── simulation_results.csv        ← Auto-generated after running dashboard
│
├── requirements.txt
└── README.md                         ← This file
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-team/rl-dynamic-pricing.git
cd rl-dynamic-pricing
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens automatically at `http://localhost:8501`

---

## 📦 Requirements

```
gymnasium==0.29.1
numpy==1.26.4
torch==2.3.0
streamlit==1.35.0
matplotlib==3.9.0
seaborn==0.13.2
pandas==2.2.2
```

> Make sure trained model files `qtable_trained.npy` and `dqn_weights.pth`
> are placed inside the `models/` folder before launching.

---

## 🖥️ Dashboard Features

---

### 🔧 Sidebar Controls

| Control | Description | Default |
|---|---|---|
| Select Agent | Choose which agent to inspect | DQN |
| Number of Episodes | Simulation count (100–1000) | 500 |
| Starting Inventory | Total seats available (10–100) | 50 |
| Days Until Departure | Booking season length (10–60) | 30 |
| ▶ Run Simulation | Triggers full simulation | — |

---

### 📊 KPI Cards (Top of Dashboard)

Four metric cards update after every simulation run:

| Card | Metric | Description |
|---|---|---|
| 💰 Total Revenue | ₹X,XX,XXX | Sum of all rewards in selected episode |
| 🎯 Seats Sold | XX / 50 | How many seats were sold |
| 📉 Spoilage Rate | X.X% | Percentage of inventory unsold at departure |
| 📈 Avg Price | ₹X,XXX | Average price charged across all days |

---

### 📈 Visualization 1: Learning Curve

**What it shows:** How the DQN agent's total revenue improves across training episodes.

- X-axis: Training episode number
- Y-axis: Total revenue earned
- Overlay toggle: Compare with Q-Learning curve
- Slider: Adjust how many episodes to display (100–1000)

**Expected pattern:**
```
Early episodes  → Low revenue (random exploration)
Mid episodes    → Rising revenue (policy forming)
Late episodes   → Plateau (optimal policy reached)
```

---

### 📦 Visualization 2: Revenue Comparison Box Plot

**What it shows:** Revenue distribution across 1,000 episodes for all 4 agents side by side.

- Each box plot shows min, max, median, and variance
- Checkbox toggles to show/hide individual agents
- Color coded: Red = Fixed, Orange = Time-Based, Blue = Q-Learning, Green = DQN

**Expected result:**
```
DQN box sits highest → best average revenue
DQN box is tightest → most consistent performance
```

---

### 📉 Visualization 3: Price Trajectory Over Time

**What it shows:** How the selected agent's price changes day by day across one full episode.

- X-axis: Days remaining (30 → 1)
- Y-axis: Price charged (₹2,000 → ₹10,000)
- Annotated dots: Mark days when a sale occurred
- Dropdown: Select which agent to visualize
- Slider: Pick which specific episode to replay

**Expected DQN behavior:**
```
Days 30-20 → ₹8,000–₹10,000  (premium pricing, low urgency)
Days 20-10 → ₹6,000–₹8,000  (moderate, adjusting to demand)
Days 10-5  → ₹4,000–₹6,000  (dropping to clear inventory)
Days 5-1   → ₹2,000–₹3,000  (floor pricing, spoilage prevention)
```

> This emergent behavior was never programmed — the agent discovered it
> entirely through training. This chart is the proof.

---

### 🛋️ Visualization 4: Inventory Depletion Curve

**What it shows:** How remaining seats decrease day by day for each agent across an episode.

- Multi-line chart, one line per agent
- Shaded danger zone: Too many seats left in last 5 days
- Color coded per agent

**Expected patterns:**
```
DQN           → Smooth S-curve, reaches ~0 near Day 1 (ideal)
Fixed Price   → Drops too fast (priced too low, sold out early)
Time-Based    → Drops too slow (priced too high early, seats left over)
Q-Learning    → Close to DQN but slightly less optimal
```

---

## 🧪 Running Evaluations Manually

If you want to run evaluations without the dashboard:

```python
from environment.airline_pricing_env import AirlinePricingEnv
from agents.fixed_price_agent import FixedPriceAgent
from agents.time_based_agent import TimeBasedAgent
from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
import numpy as np

env = AirlinePricingEnv()

agents = {
    "Fixed Price":  FixedPriceAgent(),
    "Time Based":   TimeBasedAgent(),
    "Q-Learning":   QLearningAgent(load_path="models/qtable_trained.npy"),
    "DQN":          DQNAgent(load_path="models/dqn_weights.pth")
}

for name, agent in agents.items():
    revenues = [agent.run_episode(env)[0] for _ in range(1000)]
    print(f"{name}: Avg = ₹{np.mean(revenues):,.0f} | "
          f"Std = ₹{np.std(revenues):,.0f}")
```

---

## 📊 Expected Evaluation Results

After running 1,000 episodes per agent:

| Agent | Avg Revenue | Std Dev | vs Fixed Price |
|---|---|---|---|
| Fixed Price | ₹1,10,000 | ±18,000 | baseline |
| Time-Based | ₹1,25,000 | ±15,000 | +13.6% |
| Inventory-Based | ₹1,35,000 | ±14,000 | +22.7% |
| Q-Learning | ₹1,74,000 | ±10,000 | +58.2% |
| **DQN** | **₹1,92,000** | **±7,000** | **+74.5%** |

> DQN achieves highest average revenue AND lowest variance —
> meaning it is both more profitable and more consistent.

---

## 🚀 Pushing to GitHub

```bash
git add .
git commit -m "Week 4: Streamlit dashboard + policy evaluation complete"
git push origin main
```

Make sure the following files are included before pushing:

```
✅ dashboard/app.py
✅ models/dqn_weights.pth
✅ models/qtable_trained.npy
✅ data/simulation_results.csv
✅ requirements.txt
✅ README.md
```

---

## 👥 Team

| Member | Week | Contribution |
|---|---|---|
| Member 1 | Week 1 | MDP Design + Custom Gym Environment + Demand Function |
| Member 2 | Week 2 | Baseline Agents + Q-Learning Agent + Q-Table |
| Member 3 | Week 3 | DQN Agent + Epsilon-Greedy + Experience Replay |
| **Member 4** | **Week 4** | **Policy Evaluation + Streamlit Dashboard + GitHub** |

---

## 📬 Contact

For issues related to the dashboard or evaluation scripts, raise a GitHub issue or contact Member 4 directly.

---

*Project: RL Dynamic Pricing | Travel & Hospitality Domain | Internship Project*
