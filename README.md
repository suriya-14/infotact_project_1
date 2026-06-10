# Travel & Hospitality: Reinforcement Learning for Dynamic Pricing

Welcome to the **Dynamic Pricing Agent** repository. This project aims to design and train an autonomous Reinforcement Learning (RL) agent that interacts with a simulated booking market to discover optimal dynamic pricing policies for finite, perishable inventory (such as hotel rooms or airline seats).

---

## 📊 Project Overview

Selling a fixed inventory over a limited time horizon is a classic revenue management challenge. Traditional rule-based or static systems fail to capture fluctuating demand patterns and the increasing urgency as the deadline approaches.

This project implements:
1. A custom **Gymnasium Environment** representing a stochastic booking market.
2. Three **heuristic baseline agents** (Fixed Price, Linear Discount, and Random).
3. A tabular **Q-Learning agent** for discrete state representation.
4. A **Deep Q-Network (DQN) agent** utilizing neural networks to learn policies in continuous state-spaces.

### Key Performance Indicator (KPI)
> **Goal:** The RL/DQN agent must generate **higher total episodic revenue** (cumulative reward per season) compared to all baseline strategies when evaluated over 1,000 simulated seasons.

---

## 🔄 System Architecture & Data Flow

The diagram below outlines the interaction between the pricing agent and the simulated booking market:

```mermaid
graph TD
    subgraph Market Environment (Gym Env)
        State["State: [inventory, days_left]"]
        Demand["Stochastic Demand Function"]
    end
    
    subgraph RL Agent
        Policy["DQN / Q-Table Policy"]
        Action["Select Price Level Action"]
    end

    State -->|Observation| Policy
    Policy -->|Action| Action
    Action -->|Price Choice| Demand
    Demand -->|Bookings| Reward["Reward: Price × Bookings"]
    Reward -->|Update / Experience| Policy
    Demand -->|Transition| NextState["Next State: [inventory - sold, days_left - 1]"]
    NextState --> State
```

---

## 📂 Repository Structure

```
hotel_pricing_rl/
├── environments/
│   └── hotel_pricing_env.py       # Custom Gym environment
│
├── agents/
│   ├── baseline_agents.py         # Fixed, linear discount, random
│   ├── q_learning_agent.py        # Tabular Q-Learning
│   └── dqn_agent.py               # Deep Q-Network (PyTorch)
│
├── notebooks/
│   ├── week1_env_testing.ipynb
│   ├── week2_baselines_vs_qlearning.ipynb
│   ├── week3_dqn_training.ipynb
│   └── week4_evaluation_dashboard.ipynb
│
├── scripts/
│   ├── train_dqn.py               # CLI training script
│   └── evaluate_agents.py         # Evaluation harness
│
├── plots/                         # Saved evaluation charts
├── .gitignore                     # Git ignore file (excluding data/, models/)
├── requirements.txt
└── README.md                      # Project documentation
```

---

## ⚙️ Setting Up & Running

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Training the DQN Agent
Run the command-line training script to train the DQN agent:
```bash
python scripts/train_dqn.py --episodes 10000 --save-dir models/
```

### 4. Evaluating All Agents
Run the evaluation suite to compare the DQN model against all baselines:
```bash
python scripts/evaluate_agents.py --episodes 1000
```

---

## 📄 Documentation Reference
For additional program details, please refer to:
- [PROJECT_DETAILS.md](Documents/PROJECT_DETAILS.md) — Main internship guidelines and requirements.
- [Infotact_Internship_Roadmap.md](Documents/Infotact_Internship_Roadmap.md) — Sprint-by-sprint implementation details for weeks 1-4.
- [Infotact_Project_Execution_Roadmap.md](Documents/Infotact_Project_Execution_Roadmap.md) — Team role definitions and weekly milestones.

---
*For questions or support, reach out to `support@infotact.in`*