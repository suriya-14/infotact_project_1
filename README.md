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

## 🧠 Markov Decision Process (MDP) Formulation

The dynamic pricing problem is modeled as a finite-horizon, discrete-action Markov Decision Process.

### 1. State Space ($S$)
The state is represented as a 2D continuous vector:
$$s = [\text{inventory}, \text{days\_left}]$$
- **`inventory`**: Number of remaining tickets/seats (bounds: $[0, \text{max\_inventory}]$).
- **`days_left`**: Days left until the flight departure (bounds: $[0, \text{max\_days}]$).

### 2. Action Space ($A$)
The action space is a discrete menu of price levels:
$$A \in \{0, 1, 2, 3, 4\}$$
Each action maps to a specific price:
- `0` $\rightarrow$ **₹2,000**
- `1` $\rightarrow$ **₹3,000**
- `2` $\rightarrow$ **₹4,000**
- `3` $\rightarrow$ **₹5,000**
- `4` $\rightarrow$ **₹6,000**

### 3. Transition Dynamics & Stochastic Demand
At each step, the environment computes customer demand based on the chosen price, remaining days, and the urgency as the deadline approaches.

The **expected demand** ($\lambda$) is formulated as:
$$\lambda = \text{base\_demand} \times e^{\alpha \times p} \times \left(1 + \beta \times (D - d)\right)$$

Where:
- $p$: Chosen price.
- $\alpha$: Price sensitivity parameter (`-0.0003`).
- $\beta$: Urgency factor slope (`0.05`).
- $D$: Maximum selling horizon days (`max_days`).
- $d$: Current days remaining (`days_left`).
- $D - d$: Days passed since start of the selling season.

The actual demand is sampled from a **Poisson distribution**:
$$\text{demand} \sim \text{Poisson}(\lambda)$$

The next state transitions:
$$\text{inventory}_{t+1} = \text{inventory}_t - \min(\text{demand}, \text{inventory}_t)$$
$$\text{days\_left}_{t+1} = \text{days\_left}_t - 1$$

### 4. Reward Function ($R$)
The reward is the revenue collected during the time step:
$$R_t = \text{price} \times \min(\text{demand}, \text{inventory}_t)$$

### 5. Termination Criteria
An episode terminates when:
- Remaining inventory reaches `0` (sold out).
- Remaining days reach `0` (flight departs).

---

## 📂 Repository Structure

The current repository structure on the `Manoj` branch is shown below:

```
project/
│
├── environment/
│   └── airline_pricing_env.py      ← Gymnasium environment implementation
│
├── scripts/
│   └── validate_env.py             ← CLI script to validate environment using random actions
│
├── tests/
│   └── test_env.py                 ← Unit test suite for the environment
│
├── docs/
│   └── environment_guide.md        ← Technical guide for the environment
│
├── Documents/                      ← Internship reference roadmaps and details
│   ├── Infotact_Internship_Roadmap.md
│   ├── Infotact_Project_Execution_Roadmap.md
│   └── PROJECT_DETAILS.md
│
├── .gitignore                      ← Excludes virtual env, pycache, and build files
├── requirements.txt                ← Python package dependencies
├── LICENSE                         ← MIT License
└── README.md                       ← This file
```

> [!NOTE]
> As implementation progresses over the 4-week roadmap, additional modules like `agents/` (for Fixed, Linear Discount, Q-learning, and DQN agents), `models/` (for trained policy weights), and `dashboard/` (for the Streamlit application) will be added.

---

## ⚙️ Configuration & Parameterization

The environment class is fully parameterized to allow rapid experimentation:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_inventory` | `int` | `50` | Initial inventory available to sell. |
| `max_days` | `int` | `30` | Horizon duration in days/steps. |
| `prices` | `list` | `[2000, 3000, 4000, 5000, 6000]` | Available price menu options. |
| `base_demand` | `float` | `10.0` | Base customer volume rate. |
| `price_sensitivity` | `float` | `-0.0003` | Exponential price sensitivity coefficient. |
| `urgency_factor_slope` | `float` | `0.05` | Slope factor for demand growth over time. |

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

### 3. Running Environment Validation
You can run the environment validation script, which simulates a 30-day selling season by picking random actions at each step:
```bash
python -m scripts.validate_env
```
This script validates that the environment step/reset loops are functioning properly, and outputs step-by-step logs of pricing choices, stochastic demand, seats sold, and revenue generated.

### 4. Running Unit Tests
To execute the unit test suite and verify environment logic, boundary conditions, and reproducibility:
```bash
python -m pytest
```

---

## 📄 Documentation Reference
For additional program details, please refer to:
- [docs/environment_guide.md](docs/environment_guide.md) — Technical environment details and mathematical proofs.
- [Documents/PROJECT_DETAILS.md](Documents/PROJECT_DETAILS.md) — Main internship guidelines and requirements.
- [Documents/Infotact_Internship_Roadmap.md](Documents/Infotact_Internship_Roadmap.md) — Sprint-by-sprint implementation details for weeks 1-4.
- [Documents/Infotact_Project_Execution_Roadmap.md](Documents/Infotact_Project_Execution_Roadmap.md) — Team role definitions and weekly milestones.

---
*For questions or support, reach out to `support@infotact.in`*