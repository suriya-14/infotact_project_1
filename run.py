import numpy as np
from environment.airline_pricing_env import AirlinePricingEnv
from agents.inventory_based_agent import InventoryBasedAgent

def run_simulation():
    # Initialize environment and agent
    env = AirlinePricingEnv(render_mode="human")
    agent = InventoryBasedAgent(
        prices=env.prices,
        max_inventory=env.max_inventory,
        max_days=env.max_days
    )
    
    print("=" * 70)
    print("         Starting Airline Pricing Simulation (Inventory-Based Agent)         ")
    print("=" * 70)
    
    state, info = env.reset(seed=42)
    agent.reset()
    
    done = False
    total_steps = 0
    prices_chosen = []
    demands = []
    solds = []
    bursts = 0
    
    while not done:
        action = agent.act(state)
        price = env.prices[action]
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
        
        total_steps += 1
        prices_chosen.append(price)
        demands.append(info["demand"])
        solds.append(info["sold"])
        if info["burst"] > 0:
            bursts += 1

    print("=" * 70)
    print("                             Simulation Summary                             ")
    print("=" * 70)
    print(f"Total Simulation Days:  {total_steps}")
    print(f"Total Revenue:          Rs. {info['total_revenue']:.2f}")
    print(f"Total Seats Sold:       {int(env.max_inventory - state[0])} / {env.max_inventory}")
    print(f"Remaining Seats:        {int(state[0])}")
    print(f"Average Ticket Price:   Rs. {np.mean(prices_chosen):.2f}")
    print(f"Days with Booking Bursts: {bursts}")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
