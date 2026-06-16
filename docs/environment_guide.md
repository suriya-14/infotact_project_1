# Airline Pricing Environment Guide

This guide provides technical specifications for the custom **OpenAI Gymnasium Environment** (`AirlinePricingEnv`) used for dynamic pricing reinforcement learning.

---

## 📊 Markov Decision Process (MDP) Formulation

The dynamic pricing problem is modeled as a finite-horizon, discrete-action Markov Decision Process.

### 1. State Space ($S$)
The state is represented as a 2D continuous vector (typed as `np.float32` for neural network compatibility):
$$s = [\text{inventory}, \text{days left}]$$
- **`inventory`**: Number of remaining tickets/seats (bounds: $[0, \text{max inventory}]$).
- **`days_left`**: Days left until the flight departure (bounds: $[0, \text{max days}]$).

### 2. Action Space ($A$)
The action space is a discrete menu of 20 price levels:
$$A \in \{0, 1, 2, \dots, 19\}$$
Each action maps to a specific price in ₹500 increments:
$$p(a) = 500 + (a \times 500) \quad \text{for } a \in [0, 19]$$

| Action | Price | Action | Price | Action | Price | Action | Price |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | ₹500 | 5 | ₹3,000 | 10 | ₹5,500 | 15 | ₹8,000 |
| 1 | ₹1,000 | 6 | ₹3,500 | 11 | ₹6,000 | 16 | ₹8,500 |
| 2 | ₹1,500 | 7 | ₹4,000 | 12 | ₹6,500 | 17 | ₹9,000 |
| 3 | ₹2,000 | 8 | ₹4,500 | 13 | ₹7,000 | 18 | ₹9,500 |
| 4 | ₹2,500 | 9 | ₹5,000 | 14 | ₹7,500 | 19 | ₹10,000 |

### 3. Transition Dynamics & Stochastic Demand
At each step, the environment computes customer demand using a **multi-factor demand model** that simulates realistic booking behavior.

The expected demand ($\lambda$) is formulated by combining several factors:

#### 3.1. Demand Regime Multiplier
The market operates under one of three unobservable regimes:
- **`peak`**: 1.5× multiplier
- **`normal`**: 1.0× multiplier
- **`off_peak`**: 0.6× multiplier

The regime is randomly selected at episode start. Each step, there is a 5% chance of switching to a different regime.

#### 3.2. S-Curve Urgency
Rather than a linear urgency ramp, demand follows a **sigmoid (S-curve)** over the normalized time horizon:
$$t = \frac{\text{days passed}}{\text{max days}} \quad\quad
\text{urgency} = 1.0 + \frac{\text{amplitude}}{1 + e^{-\text{steepness} \times (t - \text{midpoint})}}$$

#### 3.3. Inventory Scarcity (FOMO)
When few seats remain, demand receives an extra boost:
$$\text{scarcity} = 1.0 + \text{scarcity sensitivity} \times \left(1 - \frac{\text{inventory}}{\text{max inventory}}\right)$$

#### 3.4. Customer Segments (Price Elasticity)
Two distinct customer types respond differently to price:
- **Leisure** (70% weight, $-0.0004$ price sensitivity): Price-sensitive.
- **Business** (30% weight, $-0.0001$ price sensitivity): Price-insensitive.

$$\text{segment demand} = \text{base demand} \times \text{weight} \times \text{regime mult} \times e^{\alpha_{\text{segment}} \times p} \times \text{urgency} \times \text{scarcity}$$

The contributions are summed to form the expected baseline demand:
$$\text{expected}_{\text{segments}} = \sum_{\text{segments}} \text{segment demand}$$

#### 3.5. Market Noise
A **log-normal shock** is applied multiplicatively to simulate daily fluctuations:
$$\text{expected} = \text{expected}_{\text{segments}} \times e^{\mathcal{N}(0,\,\sigma^2)}$$

#### 3.6. Booking Bursts (Group Bookings)
With a 5% probability per step, a group booking event occurs, adding 5–15 price-insensitive bookings to the expected demand:
$$\text{expected}_{\text{total}} = \text{expected} + \text{burst}$$

#### 3.7. Final Demand Sampling
The actual demand is sampled from a **Poisson distribution**:
$$\text{demand} \sim \text{Poisson}(\text{expected}_{\text{total}})$$

The next state transitions are:
$$\text{inventory}_{t+1} = \text{inventory}_t - \min(\text{demand}, \text{inventory}_t)$$
$$\text{days left}_{t+1} = \text{days left}_t - 1$$

---

### 4. Reward Function ($R$)
The reward is the revenue collected during the time step:
$$R_t = \text{price} \times \min(\text{demand}, \text{inventory}_t)$$

### 5. Termination Criteria
An episode terminates when:
- Remaining inventory reaches `0` (sold out).
- Remaining days reach `0` (flight departs).

---

## ⚙️ Configuration & Parameterization

The environment class is fully parameterized to allow rapid experimentation:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_inventory` | `int` | `50` | Initial inventory available to sell. |
| `max_days` | `int` | `30` | Horizon duration in days/steps. |
| `prices` | `list` | `range(500, 10001, 500)` | Price menu (20 levels, ₹500–₹10,000). |
| `base_demand` | `float` | `10.0` | Base customer arrival rate per day. |
| `customer_segments` | `list[dict]` | _see below_ | Per-segment weights and price sensitivities. |
| `urgency_amplitude` | `float` | `1.0` | S-curve urgency max amplitude. |
| `urgency_steepness` | `float` | `8.0` | S-curve steepness (higher = sharper transition). |
| `urgency_midpoint` | `float` | `0.5` | S-curve midpoint (normalized time). |
| `scarcity_sensitivity` | `float` | `0.3` | How much low inventory boosts demand. |
| `demand_regimes` | `dict` | `{"peak": 1.5, "normal": 1.0, "off_peak": 0.6}` | Regime multipliers. |
| `regime_change_prob` | `float` | `0.05` | Probability of regime switch each step. |
| `market_noise_scale` | `float` | `0.1` | Log-normal noise std dev (0 = disabled). |
| `burst_probability` | `float` | `0.05` | Probability of a group booking event per step. |
| `burst_min_size` | `int` | `5` | Minimum size of a booking burst. |
| `burst_max_size` | `int` | `15` | Maximum size of a booking burst. |

Default customer segments:
```python
[
    {"name": "leisure",  "weight": 0.7, "price_sensitivity": -0.0004},
    {"name": "business", "weight": 0.3, "price_sensitivity": -0.0001},
]
```

---

## 🚀 Quick Start Example

Here is how to instantiate and run the environment in Python:

```python
import gymnasium as gym
from environment.airline_pricing_env import AirlinePricingEnv

# 1. Instantiate the environment
env = AirlinePricingEnv(max_inventory=50, max_days=30, render_mode="human")

# 2. Reset to initial state
obs, info = env.reset(seed=42)
print("Initial State:", obs)

# 3. Step through an episode
terminated = False
while not terminated:
    # Select random action index (0-19)
    action = env.action_space.sample()
    
    # Take a step
    next_obs, reward, terminated, truncated, step_info = env.step(action)
```
