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
#   - Rows    = States  → (remaining_inventory, days_until_departure)
#   - Columns = Actions → price levels ($100, $150, $200, $250, $300)
#   - Values  = Expected FUTURE revenue if we take action A in state S
#
# At each step, the agent:
#   1. Looks at current state S
#   2. Picks action A (price) using epsilon-greedy strategy
#   3. Receives reward R (revenue earned this day)
#   4. Observes new state S'
#   5. Updates Q-table using the Bellman equation:
#
#      Q(S,A) ← Q(S,A) + α * [R + γ * max(Q(S',A')) - Q(S,A)]
#
# Over thousands of episodes, Q(S,A) converges to the true optimal values.
#
# KEY HYPERPARAMETERS:
#   α (alpha)   = Learning rate       → how fast to update Q-values (0.1)
#   γ (gamma)   = Discount factor     → how much to value future rewards (0.95)
#   ε (epsilon) = Exploration rate    → probability of picking a random price
#                                       starts at 1.0, decays to 0.01
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
        q_table        (ndarray): The Q-table of shape
                                  (max_inventory+1, max_days+1, num_prices)
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
        """
        Initialize the Q-Learning agent and its Q-table.

        Args:
            price_levels  (list) : Available price actions for the agent
            max_inventory (int)  : Max inventory capacity of the environment
            max_days      (int)  : Length of the booking window
            alpha         (float): Learning rate (step size for Q updates)
            gamma         (float): Discount factor for future rewards
            epsilon       (float): Starting exploration probability (1.0 = full explore)
            epsilon_min   (float): Minimum epsilon after decay
            epsilon_decay (float): Decay multiplier applied after each episode
        """
        self.price_levels  = price_levels
        self.max_inventory = max_inventory
        self.max_days      = max_days
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Number of discrete price actions
        self.num_actions = len(price_levels)

        # Initialize Q-table with zeros
        # Shape: (inventory_states, day_states, price_actions)
        # TODO: Initialize Q-table here
        # self.q_table = np.zeros((max_inventory + 1, max_days + 1, self.num_actions))

        # Track training rewards per episode for plotting
        self.episode_rewards = []

    def select_action(self, state):
        """
        Choose a price action using the epsilon-greedy strategy.

        - With probability ε     → pick a RANDOM price (explore)
        - With probability (1-ε) → pick the BEST known price (exploit)

        Args:
            state (tuple): (remaining_inventory, days_until_departure)

        Returns:
            int  : Index into self.price_levels
            float: The actual price value
        """
        # TODO: Implement epsilon-greedy selection
        # inventory, days = state
        #
        # if np.random.rand() < self.epsilon:
        #     action_idx = np.random.randint(self.num_actions)   # explore
        # else:
        #     action_idx = np.argmax(self.q_table[inventory, days])  # exploit
        #
        # return action_idx, self.price_levels[action_idx]
        pass

    def update_q_table(self, state, action_idx, reward, next_state, done):
        """
        Apply the Bellman equation to update the Q-table.

        Bellman Update:
            Q(S,A) ← Q(S,A) + α * [R + γ * max(Q(S',A')) - Q(S,A)]

        Args:
            state      (tuple): Current (inventory, days)
            action_idx (int)  : Index of the price action taken
            reward     (float): Revenue earned this step
            next_state (tuple): Next (inventory, days) after action
            done       (bool) : Whether the episode has ended
        """
        # TODO: Implement Bellman update
        # inventory, days           = state
        # next_inventory, next_days = next_state
        #
        # current_q = self.q_table[inventory, days, action_idx]
        #
        # if done:
        #     target_q = reward
        # else:
        #     best_next_q = np.max(self.q_table[next_inventory, next_days])
        #     target_q    = reward + self.gamma * best_next_q
        #
        # # Apply Bellman update
        # self.q_table[inventory, days, action_idx] += (
        #     self.alpha * (target_q - current_q)
        # )
        pass

    def decay_epsilon(self):
        """
        Decay epsilon after each episode to shift from exploration to exploitation.
        Epsilon never goes below epsilon_min.
        """
        # TODO: Implement epsilon decay
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        pass

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
        # TODO: Implement full training loop once env is available
        # for episode in range(num_episodes):
        #     state         = env.reset()
        #     total_revenue = 0
        #     done          = False
        #
        #     while not done:
        #         action_idx, price       = self.select_action(state)
        #         next_state, reward, done, _ = env.step(price)
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
        #         print(f"Episode {episode+1}/{num_episodes} | Avg Revenue: ${avg:.2f} | ε: {self.epsilon:.3f}")
        #
        # return self.episode_rewards
        pass

    def evaluate(self, env, num_episodes=1000):
        """
        Evaluate the trained agent (epsilon=0, pure exploitation).

        Args:
            env         : AirlinePricingEnv Gym environment
            num_episodes: Number of evaluation episodes

        Returns:
            dict: mean, std, min, max revenue across all episodes
        """
        # TODO: Implement evaluation loop
        # saved_epsilon  = self.epsilon
        # self.epsilon   = 0.0        # No exploration during evaluation
        # revenues       = []
        #
        # for _ in range(num_episodes):
        #     state         = env.reset()
        #     total_revenue = 0
        #     done          = False
        #
        #     while not done:
        #         action_idx, price           = self.select_action(state)
        #         state, reward, done, _      = env.step(price)
        #         total_revenue              += reward
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
        Plot total reward per episode over training to visualize convergence.
        TODO: Implement after train() is complete.
        """
        # import matplotlib.pyplot as plt
        # plt.plot(self.episode_rewards)
        # plt.xlabel("Episode")
        # plt.ylabel("Total Revenue ($)")
        # plt.title("Q-Learning: Learning Curve")
        # plt.grid(True)
        # plt.show()
        pass

    def plot_policy(self):
        """
        Visualize the learned policy as a heatmap:
        X-axis = Days Remaining, Y-axis = Inventory, Color = Chosen Price
        TODO: Implement after Q-table has converged.
        """
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # policy = np.argmax(self.q_table, axis=2)
        # sns.heatmap(policy, cmap="YlOrRd")
        # plt.xlabel("Days Remaining")
        # plt.ylabel("Inventory Remaining")
        # plt.title("Learned Pricing Policy Heatmap")
        # plt.show()
        pass


# -----------------------------------------------------------------------------
# QUICK TEST (run this file directly to verify initialization)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = QLearningAgent()

    print("Q-Learning Agent initialized.")
    print(f"  Price levels  : {agent.price_levels}")
    print(f"  Max inventory : {agent.max_inventory}")
    print(f"  Max days      : {agent.max_days}")
    print(f"  Learning rate : {agent.alpha}")
    print(f"  Discount (γ)  : {agent.gamma}")
    print(f"  Epsilon start : {agent.epsilon}")
    print(f"  Epsilon min   : {agent.epsilon_min}")
    print(f"  Epsilon decay : {agent.epsilon_decay}")
    print()
    print("Q-table shape (when initialized):")
    print(f"  ({agent.max_inventory+1} inventory states") 
    print(f"   × {agent.max_days+1} day states")
    print(f"   × {agent.num_actions} price actions)")
    print()
    print("Status: Waiting for Member 1's environment to implement full training.")
