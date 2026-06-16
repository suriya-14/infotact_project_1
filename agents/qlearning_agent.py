# =============================================================================
# qlearning_agent.py
# Author      : Member 2
# Description : A Tabular Q-Learning agent that learns the optimal pricing
#               policy through repeated interaction with the booking environment.
#               Unlike the heuristic agents, this agent LEARNS from experience.
# =============================================================================

# -----------------------------------------------------------------------------
# HOW Q-LEARNING WORKS (Theory Summary)
# -----------------------------------------------------------------------------
# Q-Learning maintains a Q-TABLE:
#   - Rows    = States  -> (remaining_inventory, days_until_departure)
#   - Columns = Actions -> price levels ($100, $150, $200, $250, $300)
#   - Values  = Expected FUTURE revenue if we take action A in state S
#
# At each step, the agent:
#   1. Looks at current state S
#   2. Picks action A (price) using epsilon-greedy strategy
#   3. Receives reward R (revenue earned this day)
#   4. Observes new state S'
#   5. Updates Q-table using the Bellman equation:
#
#      Q(S,A) <- Q(S,A) + alpha * [R + gamma * max(Q(S',A')) - Q(S,A)]
#
# Over thousands of episodes, Q(S,A) converges to the true optimal values.
#
# KEY HYPERPARAMETERS:
#   alpha (learning rate)    = 0.1   how fast to update Q-values
#   gamma (discount factor)  = 0.95  how much to value future rewards
#   epsilon (explore rate)   = 1.0 -> decays to 0.01 over training
# -----------------------------------------------------------------------------

# TODO: Uncomment once Member 1 completes airline_pricing_env.py
# from environment.airline_pricing_env import AirlinePricingEnv

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent for airline dynamic pricing.

    Learns an optimal pricing policy by building a Q-table that maps
    every (inventory, days_left) state to the best price action.

    Attributes:
        price_levels   (list) : Discrete prices the agent can choose from
        max_inventory  (int)  : Maximum seats/rooms in the environment
        max_days       (int)  : Total days in the booking window
        alpha          (float): Learning rate
        gamma          (float): Discount factor
        epsilon        (float): Current exploration rate
        epsilon_min    (float): Minimum exploration rate (floor)
        epsilon_decay  (float): Multiplicative decay applied each episode
        q_table        (ndarray): Shape (max_inventory+1, max_days+1, num_prices)
    """

    def __init__(
        self,
        price_levels  = [100, 150, 200, 250, 300],
        max_inventory = 100,
        max_days      = 30,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
    ):
        self.price_levels  = price_levels
        self.max_inventory = max_inventory
        self.max_days      = max_days
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions   = len(price_levels)

        # Initialize Q-table with zeros
        # Shape: (inventory_states x day_states x price_actions)
        self.q_table = np.zeros((
            max_inventory + 1,   # 0 to 100 seats remaining
            max_days + 1,        # 0 to 30 days remaining
            self.num_actions     # 5 price levels
        ))

        # Track total reward per episode (used for learning curve plot)
        self.episode_rewards = []

    def select_action(self, state):
        """
        Choose a price action using the epsilon-greedy strategy.

        - With probability epsilon     -> EXPLORE: pick a random price
        - With probability 1 - epsilon -> EXPLOIT: pick best known price

        As training progresses, epsilon decays so the agent gradually
        shifts from random exploration to learned exploitation.

        Args:
            state (tuple): (remaining_inventory, days_until_departure)

        Returns:
            tuple:
                action_idx (int)  : Index into self.price_levels
                price      (float): The actual price value chosen
        """
        inventory, days = state

        # Clip state values to valid Q-table range (safety check)
        inventory = max(0, min(inventory, self.max_inventory))
        days      = max(0, min(days, self.max_days))

        if np.random.rand() < self.epsilon:
            # EXPLORE: pick a completely random price level
            action_idx = np.random.randint(self.num_actions)
        else:
            # EXPLOIT: pick the price with the highest Q-value in this state
            action_idx = np.argmax(self.q_table[inventory, days])

        return action_idx, self.price_levels[action_idx]

    def update_q_table(self, state, action_idx, reward, next_state, done):
        """
        Apply the Bellman equation to update the Q-table after each step.

        Bellman Update Rule:
            Q(S,A) <- Q(S,A) + alpha * [R + gamma * max(Q(S',A')) - Q(S,A)]

        Breaking it down:
            current_q  = Q(S,A)               what we currently believe
            best_next  = max(Q(S', all A'))    best future value from next state
            target_q   = R + gamma * best_next what we now believe it should be
            error      = target_q - current_q  how wrong we were (TD error)
            update     = alpha * error         scaled correction

        Args:
            state      (tuple): Current (inventory, days)
            action_idx (int)  : Index of the price action taken
            reward     (float): Revenue earned this step
            next_state (tuple): Next (inventory, days) after action
            done       (bool) : True if the booking season has ended
        """
        inventory, days           = state
        next_inventory, next_days = next_state

        # Clip to valid range
        inventory      = max(0, min(inventory, self.max_inventory))
        days           = max(0, min(days, self.max_days))
        next_inventory = max(0, min(next_inventory, self.max_inventory))
        next_days      = max(0, min(next_days, self.max_days))

        # Current Q-value for this state-action pair
        current_q = self.q_table[inventory, days, action_idx]

        if done:
            # No future state — target is just the immediate reward
            target_q = reward
        else:
            # Target = immediate reward + discounted best future Q-value
            best_next_q = np.max(self.q_table[next_inventory, next_days])
            target_q    = reward + self.gamma * best_next_q

        # Bellman update: move current Q-value toward the target
        td_error = target_q - current_q
        self.q_table[inventory, days, action_idx] += self.alpha * td_error

    def decay_epsilon(self):
        """
        Decay epsilon after each episode to gradually shift from
        exploration to exploitation.

        Uses multiplicative decay:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        This produces an exponential decay curve:
            Episode 0    -> epsilon = 1.000  (fully random)
            Episode 500  -> epsilon ~ 0.082
            Episode 1000 -> epsilon ~ 0.007 -> clipped to 0.01
            Episode 5000 -> epsilon = 0.01  (mostly exploiting)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=10000):
        """
        Train the Q-Learning agent over many episodes.

        Each episode = one full 30-day booking season.
        After each episode, epsilon is decayed.

        Args:
            env         : AirlinePricingEnv Gym environment (from Member 1)
            num_episodes: Number of training episodes

        Returns:
            list: Total reward per episode (for plotting the learning curve)
        """
        # TODO: Implement full training loop once env is available from Member 1
        # for episode in range(num_episodes):
        #     state         = env.reset()
        #     total_revenue = 0
        #     done          = False
        #
        #     while not done:
        #         action_idx, price            = self.select_action(state)
        #         next_state, reward, done, _  = env.step(price)
        #
        #         self.update_q_table(state, action_idx, reward, next_state, done)
        #
        #         state         = next_state
        #         total_revenue += reward
        #
        #     self.decay_epsilon()
        #     self.episode_rewards.append(total_revenue)
        #
        #     if (episode + 1) % 1000 == 0:
        #         avg = np.mean(self.episode_rewards[-1000:])
        #         print(f"Episode {episode+1}/{num_episodes} | "
        #               f"Avg Revenue: ${avg:.2f} | epsilon: {self.epsilon:.3f}")
        #
        # return self.episode_rewards
        pass

    def evaluate(self, env, num_episodes=1000):
        """
        Evaluate the trained agent with epsilon=0 (pure exploitation).

        Args:
            env         : AirlinePricingEnv Gym environment
            num_episodes: Number of evaluation episodes

        Returns:
            dict: mean, std, min, max revenue across all episodes
        """
        # TODO: Implement evaluation loop once env is available
        # saved_epsilon = self.epsilon
        # self.epsilon  = 0.0
        # revenues      = []
        #
        # for _ in range(num_episodes):
        #     state         = env.reset()
        #     total_revenue = 0
        #     done          = False
        #
        #     while not done:
        #         action_idx, price       = self.select_action(state)
        #         state, reward, done, _  = env.step(price)
        #         total_revenue          += reward
        #
        #     revenues.append(total_revenue)
        #
        # self.epsilon = saved_epsilon
        # return {
        #     "mean_revenue": np.mean(revenues),
        #     "std_revenue" : np.std(revenues),
        #     "min_revenue" : np.min(revenues),
        #     "max_revenue" : np.max(revenues),
        # }
        pass

    def plot_learning_curve(self):
        """
        Plot total reward per episode to visualize convergence.
        TODO: Implement after train() is complete.
        """
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))
        # plt.plot(self.episode_rewards, alpha=0.4, label="Per Episode")
        # moving_avg = np.convolve(self.episode_rewards,
        #                          np.ones(100)/100, mode='valid')
        # plt.plot(moving_avg, label="100-Episode Moving Avg")
        # plt.xlabel("Episode")
        # plt.ylabel("Total Revenue ($)")
        # plt.title("Q-Learning: Learning Curve")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        pass

    def plot_policy(self):
        """
        Visualize the learned policy as a heatmap.
        X = Days Remaining, Y = Inventory, Color = Chosen Price Level
        TODO: Implement after Q-table has converged.
        """
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # policy = np.argmax(self.q_table, axis=2)
        # plt.figure(figsize=(12, 6))
        # sns.heatmap(policy, cmap="YlOrRd",
        #             xticklabels=5, yticklabels=10)
        # plt.xlabel("Days Remaining")
        # plt.ylabel("Inventory Remaining")
        # plt.title("Learned Pricing Policy Heatmap")
        # plt.show()
        pass


# -----------------------------------------------------------------------------
# QUICK TEST — run this file directly to verify all implemented methods
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = QLearningAgent()

    print("=" * 50)
    print("Q-Learning Agent — Unit Test")
    print("=" * 50)

    # Test 1: Q-table shape
    print(f"\n[1] Q-table shape : {agent.q_table.shape}")
    print(f"    Expected       : (101, 31, 5)")
    assert agent.q_table.shape == (101, 31, 5), "Q-table shape mismatch!"
    print("    PASSED")

    # Test 2: select_action — explore mode (epsilon = 1.0)
    print(f"\n[2] select_action() — explore mode (epsilon=1.0)")
    agent.epsilon = 1.0
    action_idx, price = agent.select_action((50, 15))
    print(f"    Chosen price   : ${price} (index {action_idx})")
    assert price in agent.price_levels, "Price not in valid price levels!"
    print("    PASSED")

    # Test 3: select_action — exploit mode (epsilon = 0.0)
    print(f"\n[3] select_action() — exploit mode (epsilon=0.0)")
    agent.epsilon = 0.0
    # Manually set a high Q-value for price $250 (index 3) at state (50, 15)
    agent.q_table[50, 15, 3] = 9999
    action_idx, price = agent.select_action((50, 15))
    print(f"    Chosen price   : ${price} (index {action_idx})")
    assert price == 250, f"Expected $250 but got ${price}!"
    print("    PASSED")

    # Test 4: update_q_table (Bellman equation)
    print(f"\n[4] update_q_table() — Bellman update")
    agent.q_table = np.zeros((101, 31, 5))   # reset Q-table
    agent.update_q_table(
        state      = (50, 15),
        action_idx = 2,          # price $200
        reward     = 400.0,      # earned $400 this step
        next_state = (49, 14),
        done       = False
    )
    updated_val = agent.q_table[50, 15, 2]
    print(f"    Q[50,15,2] after update : {updated_val:.4f}")
    assert updated_val > 0, "Q-value should be > 0 after positive reward!"
    print("    PASSED")

    # Test 5: decay_epsilon
    print(f"\n[5] decay_epsilon()")
    agent.epsilon = 1.0
    for _ in range(1000):
        agent.decay_epsilon()
    print(f"    Epsilon after 1000 decays : {agent.epsilon:.4f}")
    assert agent.epsilon >= agent.epsilon_min, "Epsilon went below minimum!"
    print("    PASSED")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("Waiting for Member 1's env to implement train() and evaluate()")
    print("=" * 50)
