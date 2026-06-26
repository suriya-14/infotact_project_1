# ✈️ RL Dynamic Pricing – Intelligent Airline Ticket Pricing using Reinforcement Learning

## 📌 Project Overview

Dynamic pricing is one of the most important applications of Artificial Intelligence in the airline industry. Airlines continuously change ticket prices based on customer demand, remaining seats, and the number of days left before departure. Traditional pricing strategies rely on manually designed rules, which often fail to maximize revenue under changing market conditions.

This project develops an **Intelligent Airline Ticket Pricing System** using **Reinforcement Learning (RL)**. Instead of following predefined pricing rules, an RL agent interacts with a simulated airline booking environment, learns from experience, and discovers pricing strategies that maximize total revenue over the booking period.

The project compares **five different pricing agents**, ranging from simple rule-based methods to advanced Deep Reinforcement Learning, allowing us to evaluate different approaches to dynamic pricing.

---

# 🎯 Objectives

* Build a realistic airline ticket pricing simulator.
* Implement multiple pricing strategies.
* Train reinforcement learning agents to maximize revenue.
* Compare rule-based and learning-based pricing methods.
* Visualize agent performance using an interactive dashboard.

---

# 🏗 System Workflow

```
Customer Booking Environment
            │
            ▼
Environment provides State
(days remaining, inventory remaining)

            │
            ▼
Agent selects Ticket Price

            │
            ▼
Environment simulates Demand

            │
            ▼
Tickets are Sold

            │
            ▼
Revenue becomes Reward

            │
            ▼
Agent learns better pricing strategy
```

---

# 🌍 Environment

**File:** `environment/airline_pricing_env.py`

The custom Gymnasium environment simulates the airline booking process.

### State Space

The environment provides two state variables:

```
State =

[
 Days Remaining,
 Inventory Remaining
]
```

Example:

```
[20, 35]

20 days before departure
35 seats remaining
```

---

### Action Space

The agent selects one of **20 ticket prices**.

Price range:

```
₹500
↓

₹10000
```

---

### Demand Model

Demand depends on:

* Ticket Price
* Remaining Days
* Random customer behaviour

Lower prices generally increase demand while higher prices reduce demand.

---

### Reward Function

```
Reward = Ticket Price × Tickets Sold
```

The objective of every agent is to maximize the total reward (revenue).

---

### Episode Ends When

* All seats are sold

OR

* Flight departure day arrives

---

# 🤖 Pricing Agents

## 1. Fixed Price Agent

**Type:** Rule-Based

This agent always charges the same ticket price throughout the booking window.

Example:

```
Day 30 → ₹3000

Day 20 → ₹3000

Day 10 → ₹3000

Day 1 → ₹3000
```

### Advantages

* Very simple
* No training required
* Fast execution

### Limitations

* Cannot adapt to demand
* Ignores time
* Ignores inventory

---

## 2. Time-Based Agent

**Type:** Rule-Based

The ticket price changes only according to the number of remaining booking days.

Example:

```
30 Days Left → High Price

15 Days Left → Medium Price

5 Days Left → Low Price
```

### Advantages

* Mimics traditional airline pricing
* Easy to understand

### Limitations

* Does not consider remaining seats

---

## 3. Q-Learning Agent

**Type:** Reinforcement Learning

The Q-Learning agent learns from repeated interaction with the environment.

It stores knowledge inside a Q-table.

```
State

↓

Action

↓

Expected Future Reward
```

The Q-table is updated using the Bellman Equation.

```
Q(s,a)=Q(s,a)+α[r+γmaxQ(s',a')−Q(s,a)]
```

### Advantages

* Learns automatically
* Improves with experience

### Limitations

* Q-table becomes large for complex environments

---

## 4. Inventory-Based Agent

**Type:** Rule-Based Heuristic

This agent considers both:

* Remaining inventory
* Remaining booking days

Pricing decisions are based on:

### Scarcity

Few seats remaining

↓

Increase price

### Urgency

Departure is close

Many seats remaining

↓

Reduce price

This produces more realistic pricing behaviour than simple rule-based agents.

---

## 5. Deep Q-Network (DQN)

**Type:** Deep Reinforcement Learning

The DQN Agent replaces the Q-table with a neural network.

Architecture:

```
Input Layer

2 Neurons

↓

Hidden Layer

64 Neurons

↓

Hidden Layer

64 Neurons

↓

Output Layer

20 Q Values
```

The DQN agent includes:

* Replay Buffer
* Target Network
* Neural Network
* Experience Replay

This enables the agent to generalize to unseen states and achieve higher revenue.

---

# 📊 Dashboard

**Framework:** Streamlit

The dashboard allows users to:

* Select any pricing agent
* Run booking simulations
* View pricing decisions
* Compare revenues
* Display KPI cards
* Visualize ticket prices
* Compare all five agents

Run using:

```bash
streamlit run dashboard/app.py
```

---

# 📁 Project Structure

```
rl-dynamic-pricing/

│

├── environment/

│   └── airline_pricing_env.py

│

├── agents/

│   ├── fixed_price_agent.py

│   ├── time_based_agent.py

│   ├── qlearning_agent.py

│   ├── inventory_based_agent.py

│   └── dqn_agent.py

│

├── models/

│   └── dqn_weights.pth

│

├── dashboard/

│   └── app.py

│

└── README.md
```

---

# 👥 Team Contributions

## Member 1

* Designed the Gymnasium environment
* Implemented the demand model
* Developed the Fixed Price Agent

---

## Member 2

* Developed the Time-Based Agent
* Implemented the Q-Learning Agent
* Performed tabular RL training

---

## Member 3

* Developed the Inventory-Based Agent
* Implemented the DQN Agent
* Built the Replay Buffer
* Designed the Neural Network
* Generated trained DQN model weights

---

## Member 4

* Developed the Streamlit dashboard
* Created interactive charts
* Implemented KPI cards
* Built simulation comparison interface

---

# 📈 Agent Comparison

| Agent           | Uses Time | Uses Inventory | Learns | Complexity |
| --------------- | --------- | -------------- | ------ | ---------- |
| Fixed Price     | ❌         | ❌              | ❌      | Very Low   |
| Time-Based      | ✅         | ❌              | ❌      | Low        |
| Q-Learning      | ✅         | ✅              | ✅      | Medium     |
| Inventory-Based | ✅         | ✅              | ❌      | Medium     |
| DQN             | ✅         | ✅              | ✅      | High       |

---

# 🛠 Technology Stack

* Python
* Gymnasium
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Streamlit

---

# 🚀 Future Scope

* Real airline datasets
* Multiple flight scheduling
* Competitor pricing
* Weather-based demand prediction
* Holiday and festival demand modelling
* Multi-Agent Reinforcement Learning
* Cloud deployment

---

# ✅ Conclusion

This project demonstrates how Reinforcement Learning can be applied to solve real-world airline ticket pricing problems. By comparing rule-based and learning-based approaches, we show that intelligent agents can learn pricing strategies that adapt to changing demand, remaining inventory, and booking time. Among all implemented methods, the Deep Q-Network (DQN) provides the highest potential for maximizing airline revenue and serves as the most advanced solution in this project.
