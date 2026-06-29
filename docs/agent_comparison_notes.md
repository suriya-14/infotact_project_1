# 📊 Agent Comparison Notes — Q-Learning vs DQN

> **Author:** Member 2
> **Date:** June 29, 2026
> **Purpose:** Compare Member 2's Q-Learning agent with
> Member 3's DQN agent for final review preparation

---

## Why Two Different RL Approaches?

| Aspect | Q-Learning (Member 2) | DQN (Member 3) |
|--------|----------------------|-----------------|
| Type | Tabular RL | Deep RL |
| State representation | Bucketed (10×10) | Continuous |
| Learning model | Q-table | Neural Network |
| Training speed | Fast | Slower |
| State space limit | Small-medium | Large/continuous |
| Interpretability | High (table readable) | Low (black box) |
| Best for | Discrete, small state spaces | Complex, large state spaces |

---

## Q-Learning (Member 2) — Summary

```
Architecture : Q-table (10 × 10 × 20 = 2,000 values)
Training     : 20,000 episodes
Alpha        : 0.2
Gamma        : 0.99
Epsilon decay: 0.9995
Init         : Optimistic (200,000)
```

### Results
| Metric | Value |
|--------|-------|
| Mean Revenue | Rs 4,50,000 |
| Improvement over baseline | +46.60% |
| Convergence | ~Episode 10,000 |

---

## DQN (Member 3) — Expected Advantages

- Handles continuous state space without bucketing
- Neural network can capture complex pricing patterns
- Experience replay stabilizes training
- Expected to outperform tabular Q-Learning

---

## Expected Final Comparison Table

| Agent | Mean Revenue | vs Time-Based |
|-------|-------------|---------------|
| Fixed Price (Member 1) | Rs ~2,00,000 | baseline |
| Time-Based (Member 2) | Rs 2,96,545 | — |
| Q-Learning (Member 2) | Rs 4,50,000 | +46.60% |
| DQN (Member 3) | TBD | TBD |

---

## Key Insight

> Q-Learning proves that even a simple tabular RL approach
> significantly outperforms heuristic baselines (+46.60%).
> DQN is expected to improve further by handling the full
> continuous state space without bucketing approximation.

---

## What This Means for the Project

- ✅ Q-Learning is a strong, interpretable baseline RL agent
- ✅ The bucketing approach made convergence practical
- ✅ Results validate the RL approach over rule-based pricing
- 🔜 DQN comparison to be added after Member 3 shares results
