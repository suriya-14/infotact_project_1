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
The action space is a discrete menu of 20 price levels:
$$A \in \{0, 1, 2, \dots, 19\}$$
Each action maps to a price in ₹500 increments:
$$p(a) = 500 + (a \times 500) \quad \text{for } a \in [0, 19]$$

| Action | Price | Action | Price | Action | Price | Action | Price |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | ₹500 | 5 | ₹3,000 | 10 | ₹5,500 | 15 | ₹8,000 |
| 1 | ₹1,000 | 6 | ₹3,500 | 11 | ₹6,000 | 16 | ₹8,500 |
| 2 | ₹1,500 | 7 | ₹4,000 | 12 | ₹6,500 | 17 | ₹9,000 |
| 3 | ₹2,000 | 8 | ₹4,500 | 13 | ₹7,000 | 18 | ₹9,500 |
| 4 | ₹2,500 | 9 | ₹5,000 | 14 | ₹7,500 | 19 | ₹10,000 |

### 3. Transition Dynamics & Stochastic Demand

At each step, the environment computes customer demand using a **multi-factor demand model** that simulates realistic booking behavior. The next state transitions as:

$$\text{inventory}_{t+1} = \text{inventory}_t - \min(\text{demand}, \text{inventory}_t)$$
$$\text{days\_left}_{t+1} = \text{days\_left}_t - 1$$

The demand model combines **6 distinct features** that interact multiplicatively (except bursts which are additive). Below is the complete computation pipeline:

---

#### 3a. Demand Regime Multiplier

The market operates under one of three unobservable regimes that scale overall demand:

| Regime | Multiplier | Description |
|:---|:---:|:---|
| `peak` | 1.5× | High season — strong demand |
| `normal` | 1.0× | Baseline demand |
| `off_peak` | 0.6× | Low season — weak demand |

The regime is randomly selected at episode start. Each step, there is a 5% chance of switching to a _different_ regime (never stays on the same one). Since the regime is **not included in the state**, the agent must infer it from the reward signal.

---

#### 3b. Customer Segments (Price Elasticity)

Two distinct customer types respond differently to price:

| Segment | Weight | Price Sensitivity | Behavior |
|:---|:---:|:---:|:---|
| **Leisure** | 70% | −0.0004 | Price-sensitive — demand drops sharply as price rises |
| **Business** | 30% | −0.0001 | Price-insensitive — willing to pay high fares |

Each segment computes its own demand contribution:

$$\text{segment\_demand} = \text{base\_demand} \times \text{weight} \times \text{regime\_mult} \times e^{\alpha_{\text{segment}} \times p}$$

The contributions are summed to form the total expected demand:

$$\text{expected}_{\text{segments}} = \sum_{\text{segments}} \text{segment\_demand}$$

---

#### 3c. S-Curve Urgency

Rather than a linear urgency ramp, demand follows a **sigmoid (S-curve)** over the normalized time horizon:

$$t = \frac{\text{days\_passed}}{\text{max\_days}} \quad\quad
\text{urgency} = 1.0 + \frac{\text{amplitude}}{1 + e^{-\text{steepness} \times (t - \text{midpoint})}}$$

This produces three phases:
1. **Early days** ($t < 0.3$): urgency is low and flat — customers are not in a hurry.
2. **Mid-horizon** ($t \approx 0.5$): urgency rises rapidly as departure approaches.
3. **Near departure** ($t > 0.7$): urgency plateaus — the most anxious customers have already booked.

Default parameters: `amplitude=1.0`, `steepness=8.0`, `midpoint=0.5`.

---

#### 3d. Inventory Scarcity (FOMO)

When few seats remain, demand receives an extra boost simulating customer panic:

$$\text{scarcity} = 1.0 + \text{scarcity\_sensitivity} \times \left(1 - \frac{\text{inventory}}{\text{max\_inventory}}\right)$$

| Inventory Remaining | Scarcity Factor |
|:---:|:---:|
| 50 / 50 (full) | 1.00× (no effect) |
| 25 / 50 | 1.15× |
| 10 / 50 | 1.24× |
| 1 / 50 | 1.29× |

Default `scarcity_sensitivity=0.3`.

---

#### 3e. Market Noise

A **log-normal shock** is applied multiplicatively each step to simulate unpredictable market fluctuations:

$$\text{noise} = e^{\mathcal{N}(0,\,\sigma^2)} \quad\quad
\text{expected} = \text{expected} \times \text{noise}$$

Where $\sigma = 0.1$ by default. This produces random daily variations of roughly ±10%.

---

#### 3f. Booking Bursts (Group Bookings)

With a 5% probability per step, a **group booking** event occurs, adding 5–15 price-insensitive passengers to the expected demand:

$$\text{burst} \sim \text{Uniform}\{5, 6, \dots, 15\} \quad\quad
\text{expected} = \text{expected} + \text{burst}$$

Bursts are tracked in the info dict as `info["burst"]` and are independent of the chosen price.

---

#### 3g. Final Demand Sampling

The total expected demand after all factors is sampled from a **Poisson distribution** and clipped to remaining inventory:

$$\text{expected}_{\text{total}} = \text{expected}_{\text{segments}} \times \text{urgency} \times \text{scarcity} \times \text{noise} + \text{burst}$$

$$\text{demand} = \min\bigl(\text{Poisson}(\text{expected}_{\text{total}}),\; \text{inventory}\bigr)$$

---

#### 3h. Complete Demand Formula

Putting it all together:

$$\begin{aligned}
\lambda &= \text{base\_demand} \times \text{regime\_mult} \times \text{urgency}(t) \times \text{scarcity}(\text{inv}) \times \text{noise} \\
&\quad \times \bigl[ w_{\text{leisure}} \, e^{\alpha_{\text{leisure}} p} + w_{\text{business}} \, e^{\alpha_{\text{business}} p} \bigr] + \text{burst}
\end{aligned}$$

$$
\text{bookings} \sim \text{Poisson}(\lambda) \quad\quad
\text{sold} = \min(\text{bookings}, \text{inventory})
$$

### 4. Reward Function ($R$)
The reward is the revenue collected during the time step:
$$R_t = \text{price} \times \min(\text{demand}, \text{inventory}_t)$$

### 5. Termination Criteria
An episode terminates when:
- Remaining inventory reaches `0` (sold out).
- Remaining days reach `0` (flight departs).

Maximum episode length = `max_days` (default 30) steps.

---

### 6. Info Dict

Each call to `step()` returns an `info` dictionary with detailed diagnostics:

| Key | Type | Description |
|:---|:---:|:---|
| `total_revenue` | `float` | Cumulative revenue earned so far in the episode |
| `sold` | `int` | Seats sold in this step |
| `demand` | `int` | Raw demand before inventory clip |
| `price` | `int` | Price chosen at this step |
| `regime` | `str` | Current demand regime (`peak` / `normal` / `off_peak`) |
| `burst` | `int` | Size of group booking burst (0 if none) |

---

### 7. Rendering

The environment supports two render modes:

- **`"human"`**: prints a status line to stdout each step
- **`"ansi"`**: returns the status line as a string (for logging)

Example output:
```
Day  2/30 | Inventory: 44/50 | Regime:     peak | Revenue: 18000.00
```

The line shows: current day, remaining/max inventory, active demand regime, and cumulative revenue.

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
| `max_inventory` | `int` | `50` | Initial inventory available to sell |
| `max_days` | `int` | `30` | Horizon duration in days/steps |
| `prices` | `list` | `range(500, 10001, 500)` | Price menu (20 levels, ₹500–₹10,000) |
| `base_demand` | `float` | `10.0` | Base customer arrival rate per day |
| `customer_segments` | `list[dict]` | _see below_ | Per-segment weights and price sensitivities |
| `urgency_amplitude` | `float` | `1.0` | S-curve urgency max amplitude |
| `urgency_steepness` | `float` | `8.0` | S-curve steepness (higher = sharper transition) |
| `urgency_midpoint` | `float` | `0.5` | S-curve midpoint (normalized time) |
| `scarcity_sensitivity` | `float` | `0.3` | How much low inventory boosts demand |
| `demand_regimes` | `dict` | `{peak: 1.5, normal: 1.0, off_peak: 0.6}` | Regime multipliers |
| `regime_change_prob` | `float` | `0.05` | Probability of regime switch each step |
| `market_noise_scale` | `float` | `0.1` | Log-normal noise std dev (0 = disabled) |
| `burst_probability` | `float` | `0.05` | Probability of a group booking event per step |
| `burst_min_size` | `int` | `5` | Minimum size of a booking burst |
| `burst_max_size` | `int` | `15` | Maximum size of a booking burst |

Default customer segments:
```python
[
    {"name": "leisure",  "weight": 0.7, "price_sensitivity": -0.0004},
    {"name": "business", "weight": 0.3, "price_sensitivity": -0.0001},
]
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