import pytest
import numpy as np
import gymnasium as gym
from environment.airline_pricing_env import AirlinePricingEnv


def test_env_initialization():
    """Test that the environment initializes with correct default spaces and values."""
    env = AirlinePricingEnv()
    assert env.max_inventory == 50
    assert env.max_days == 30
    assert len(env.prices) == 5
    assert env.action_space.n == 5
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert np.all(env.observation_space.low == np.array([0, 0]))
    assert np.all(env.observation_space.high == np.array([50, 30]))


def test_env_parameterization():
    """Test that the environment handles customized parameters correctly."""
    custom_prices = [1000, 1500, 2000]
    env = AirlinePricingEnv(
        max_inventory=10,
        max_days=5,
        prices=custom_prices,
        base_demand=5.0,
        price_sensitivity=-0.001,
        urgency_factor_slope=0.1,
    )
    assert env.max_inventory == 10
    assert env.max_days == 5
    assert env.prices == custom_prices
    assert env.action_space.n == 3
    assert np.all(env.observation_space.high == np.array([10, 5]))
    assert env.base_demand == 5.0
    assert env.price_sensitivity == -0.001
    assert env.urgency_factor_slope == 0.1


def test_env_reset():
    """Test environment reset returns correct observation shape and info dict."""
    env = AirlinePricingEnv(max_inventory=40, max_days=20)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert obs[0] == 40.0
    assert obs[1] == 20.0
    assert isinstance(info, dict)
    assert len(info) == 0


def test_env_step_validity():
    """Test that taking a step updates states correctly and outputs expected types."""
    env = AirlinePricingEnv(max_inventory=50, max_days=30)
    obs, info = env.reset(seed=100)
    
    # Take action 0 (price level 1: ₹2000)
    next_obs, reward, terminated, truncated, step_info = env.step(0)
    
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (2,)
    assert next_obs[1] == 29.0  # days remaining decrements by 1
    assert next_obs[0] <= 50.0  # inventory decrements or stays same
    
    assert isinstance(reward, float)
    assert reward >= 0.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not truncated
    assert isinstance(step_info, dict)
    assert "total_revenue" in step_info
    assert "sold" in step_info
    assert "demand" in step_info
    assert step_info["price"] == env.prices[0]
    assert reward == float(env.prices[0] * step_info["sold"])


def test_env_seeding():
    """Test that setting a seed guarantees identical demand sequences (reproducibility)."""
    env1 = AirlinePricingEnv(max_inventory=100, max_days=100)
    env2 = AirlinePricingEnv(max_inventory=100, max_days=100)

    # Use the same seed
    env1.reset(seed=12345)
    env2.reset(seed=12345)

    for _ in range(10):
        action = 2  # fixed price level index
        obs1, reward1, done1, _, info1 = env1.step(action)
        obs2, reward2, done2, _, info2 = env2.step(action)
        assert info1["sold"] == info2["sold"]
        assert reward1 == reward2
        assert np.array_equal(obs1, obs2)
        assert done1 == done2


def test_env_termination_by_days():
    """Test that environment terminates when selling days run out."""
    env = AirlinePricingEnv(max_inventory=100, max_days=2)
    env.reset(seed=42)

    # Day 1 step
    _, _, terminated, _, _ = env.step(2)
    assert not terminated

    # Day 2 step (last day)
    _, _, terminated, _, _ = env.step(2)
    assert terminated


def test_env_termination_by_inventory():
    """Test that environment terminates immediately if inventory drops to zero."""
    # Using high demand, low inventory to guarantee sellout on first step
    env = AirlinePricingEnv(max_inventory=1, max_days=10, base_demand=1000.0)
    env.reset(seed=42)

    # Step should sell the 1 remaining item and terminate
    obs, reward, terminated, _, info = env.step(0)
    assert obs[0] == 0.0
    assert terminated
    assert info["sold"] == 1
    assert reward == env.prices[0]


def test_invalid_action():
    """Test that the environment raises an error for invalid actions."""
    env = AirlinePricingEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(5)  # Action space is size 5 (0 to 4)
    with pytest.raises(ValueError):
        env.step(-1)
