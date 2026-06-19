# =============================================================================
# test_env.py
# Author      : Member 2
# Description : Quick test script to verify that Member 1's
#               AirlinePricingEnv works correctly before connecting
#               it to the Q-Learning and Time-Based agents.
# Run         : python test_env.py
# =============================================================================

from environment.airline_pricing_env import AirlinePricingEnv

def test_reset():
    """Test that env.reset() returns a valid initial state."""
    env   = AirlinePricingEnv()
    state = env.reset()

    print("=" * 45)
    print("TEST 1 — env.reset()")
    print("=" * 45)
    print(f"  Initial state     : {state}")
    print(f"  Inventory         : {state[0]}")
    print(f"  Days remaining    : {state[1]}")
    assert state is not None, "State should not be None!"
    assert state[0] > 0,      "Inventory should be > 0 at start!"
    assert state[1] > 0,      "Days should be > 0 at start!"
    print("  PASSED ✅")


def test_step():
    """Test that env.step() returns correct values for a given price."""
    env   = AirlinePricingEnv()
    state = env.reset()

    print()
    print("=" * 45)
    print("TEST 2 — env.step(price=200)")
    print("=" * 45)

    next_state, reward, done, info = env.step(200)

    print(f"  Next state        : {next_state}")
    print(f"  Reward            : {reward}")
    print(f"  Done              : {done}")
    print(f"  Info              : {info}")
    assert next_state is not None,  "Next state should not be None!"
    assert reward >= 0,             "Reward should be non-negative!"
    assert isinstance(done, bool),  "Done should be a boolean!"
    print("  PASSED ✅")


def test_full_episode():
    """Run one complete episode and print total revenue."""
    env           = AirlinePricingEnv()
    state         = env.reset()
    done          = False
    total_revenue = 0
    step_count    = 0

    print()
    print("=" * 45)
    print("TEST 3 — Full Episode (price=$200 fixed)")
    print("=" * 45)

    while not done:
        next_state, reward, done, info = env.step(200)
        total_revenue += reward
        step_count    += 1
        state          = next_state

    print(f"  Steps completed   : {step_count}")
    print(f"  Total Revenue     : ${total_revenue:.2f}")
    print(f"  Final state       : {state}")
    assert step_count > 0,      "Episode should have at least 1 step!"
    assert total_revenue >= 0,  "Total revenue should be non-negative!"
    print("  PASSED ✅")


def test_multiple_prices():
    """Test environment response to different price levels."""
    env    = AirlinePricingEnv()
    prices = [100, 150, 200, 250, 300]

    print()
    print("=" * 45)
    print("TEST 4 — Different Price Levels")
    print("=" * 45)

    for price in prices:
        state              = env.reset()
        next_state, reward, done, info = env.step(price)
        print(f"  Price ${price} -> reward: ${reward:.2f} | next state: {next_state}")

    print("  PASSED ✅")


# -----------------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("Starting Environment Tests...")
    print()

    try:
        test_reset()
        test_step()
        test_full_episode()
        test_multiple_prices()

        print()
        print("=" * 45)
        print("ALL TESTS PASSED ✅")
        print("Environment is ready to connect with agents!")
        print("=" * 45)

    except AssertionError as e:
        print(f"\nTEST FAILED ❌: {e}")
    except Exception as e:
        print(f"\nERROR ❌: {e}")
        print("Check that airline_pricing_env.py is in environment/ folder")
