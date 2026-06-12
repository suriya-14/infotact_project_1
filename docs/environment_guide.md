# Airline Pricing Environment Guide

This guide provides technical specifications for the custom **OpenAI Gymnasium Environment** (`AirlinePricingEnv`) used for dynamic pricing reinforcement learning.

---

## 📊 Markov Decision Process (MDP) Formulation

The dynamic pricing problem is modeled as a finite-horizon, discrete-action Markov Decision Process.

### 1. State Space ($S$)
The state is represented as a 2D continuous vector (typed as `np.float32` for neural network compatibility):
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
    # Select random action index (0-4)
    action = env.action_space.sample()
    
    # Take a step
    next_obs, reward, terminated, truncated, step_info = env.step(action)
```
