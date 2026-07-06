import numpy as np
import pytest
from environment.airline_pricing_env import AirlinePricingEnv
from agents.baseline_agents import FixedPriceAgent, LinearDiscountAgent, RandomAgent


def test_fixed_price_agent():
    """Test that FixedPriceAgent returns the correct action index."""
    agent_default = FixedPriceAgent()
    obs = np.array([50.0, 30.0], dtype=np.float32)

    # Test default action index (7)
    assert agent_default(obs) == 7
    assert agent_default.act(obs) == 7

    # Test custom action index (10)
    agent_custom = FixedPriceAgent(action_index=10)
    assert agent_custom(obs) == 10
    assert agent_custom.act(obs) == 10


def test_linear_discount_agent():
    """Test that LinearDiscountAgent decreases price over days and maps correctly."""
    prices = [1000, 2000, 3000, 4000, 5000]
    agent = LinearDiscountAgent(
        start_price=5000, daily_drop=500, max_days=10, prices=prices
    )

    # Day 0 (9 days left): target price = 5000 - 500*0 = 5000 -> index 4
    obs = np.array([10.0, 10.0], dtype=np.float32)
    assert agent(obs) == 4

    # Day 2 (8 days left): target price = 5000 - 500*2 = 4000 -> index 3
    obs = np.array([10.0, 8.0], dtype=np.float32)
    assert agent(obs) == 3

    # Day 5 (5 days left): target price = 5000 - 500*5 = 2500 -> closest is 2000 (index 1) or 3000 (index 2)
    # Target price 2500 is equidistant from 2000 and 3000. argmin chooses the first one (index 1)
    obs = np.array([10.0, 5.0], dtype=np.float32)
    assert agent(obs) in (1, 2)


def test_random_agent():
    """Test that RandomAgent returns actions within valid bounds."""
    agent = RandomAgent(action_space_size=20)
    obs = np.array([50.0, 30.0], dtype=np.float32)

    actions = [agent(obs) for _ in range(100)]
    assert all(0 <= a < 20 for a in actions)
    assert len(set(actions)) > 1  # Should have some variation


def test_agent_runs_in_env():
    """Test that FixedPriceAgent can run an episode to completion in the environment."""
    env = AirlinePricingEnv(max_inventory=10, max_days=5)
    agent = FixedPriceAgent(action_index=7)

    obs, info = env.reset(seed=123)
    assert env.inventory == 10
    assert env.days_left == 5

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        action = agent(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step_count += 1

    assert step_count <= 5
    assert env.inventory <= 10
    assert env.days_left == 0 or env.inventory == 0
