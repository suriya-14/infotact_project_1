import argparse
from environment.airline_pricing_env import AirlinePricingEnv


def main():
    parser = argparse.ArgumentParser(description="Validate the Airline Pricing Gym environment.")
    parser.add_argument("--inventory", type=int, default=50, help="Initial inventory (default: 50)")
    parser.add_argument("--days", type=int, default=30, help="Total selling days (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"=== Initializing AirlinePricingEnv with seed={args.seed} ===")
    print(f"Max Inventory: {args.inventory}, Max Days: {args.days}\n")

    env = AirlinePricingEnv(
        max_inventory=args.inventory,
        max_days=args.days,
        render_mode="human",
    )

    obs, info = env.reset(seed=args.seed)
    print(f"Initial State: Inventory={obs[0]:.0f}, Days Remaining={obs[1]:.0f}\n")

    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_count += 1
        # Sample a random action (0 to 4)
        action = env.action_space.sample()
        price = env.prices[action]

        obs, reward, terminated, truncated, step_info = env.step(action)

        print(
            f"Step {step_count:2d} | "
            f"Action Index: {action} (Price: ₹{price:4d}) | "
            f"Demand: {step_info['demand']:2d} | "
            f"Sold: {step_info['sold']:2d} | "
            f"Revenue: ₹{reward:6.0f} | "
            f"Remaining Inventory: {obs[0]:2.0f} | "
            f"Days Left: {obs[1]:2.0f}"
        )

    print("\n=== Episode Finished ===")
    print(f"Total Steps Run: {step_count}")
    print(f"Total Episodic Revenue: ₹{env.total_revenue:.2f}")
    print(f"Final Inventory: {env.inventory}/{env.max_inventory}")
    print(f"Reason for termination: {'Out of Inventory' if env.inventory <= 0 else 'Out of Time'}")


if __name__ == "__main__":
    main()
