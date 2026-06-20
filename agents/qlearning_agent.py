# =============================================================================
# qlearning_agent.py
# Author      : Member 2
# Description : Tabular Q-Learning agent for airline dynamic pricing.
#               Learns optimal pricing policy through environment interaction.
#               Outperforms heuristic baselines by learning from experience.
# =============================================================================

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.airline_pricing_env import AirlinePricingEnv


class QLearningAgent:
    """
    Tabular Q-Learning agent for airline dynamic pricing.

    Q-TABLE shape: (max_inventory+1, max_days+1, num_price_levels)
        - Rows    = inventory states  (0 to 50)
        - Columns = day states        (0 to 30)
        - Depth   = price actions     (0 to 19)
        - Values  = expected future revenue for taking that action

    Bellman Update:
        Q(S,A) <- Q(S,A) + alpha * [R + gamma * max(Q(S',A')) - Q(S,A)]
    """

    def __init__(
        self,
        max_inventory = 50,
        max_days      = 30,
        num_actions   = 20,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
    ):
        """
        Args:
            max_inventory (int)  : Max seats (matches env: 50)
            max_days      (int)  : Booking window (matches env: 30)
            num_actions   (int)  : Number of price levels (matches env: 20)
            alpha         (float): Learning rate
            gamma         (float): Discount factor
            epsilon       (float): Starting exploration rate
            epsilon_min   (float): Minimum exploration rate
            epsilon_decay (float): Decay per episode
        """
        self.max_inventory = max_inventory
        self.max_days      = max_days
        self.num_actions   = num_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape (51, 31, 20) — all zeros at start
        self.q_table = np.zeros((
            max_inventory + 1,
            max_days + 1,
            num_actions
        ))

        # Track rewards for plotting learning curve
        self.episode_rewards = []

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state (np.array): [inventory, days_remaining] from env

        Returns:
            int: price action index (0-19)
        """
        inventory = int(np.clip(state[0], 0, self.max_inventory))
        days      = int(np.clip(state[1], 0, self.max_days))

        if np.random.rand() < self.epsilon:
            # EXPLORE: random price
            return np.random.randint(self.num_actions)
        else:
            # EXPLOIT: best known price
            return int(np.argmax(self.q_table[inventory, days]))

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Apply Bellman equation to update Q-table.

        Args:
            state      (np.array): Current [inventory, days]
            action     (int)     : Price index taken
            reward     (float)   : Revenue earned this step
            next_state (np.array): Next [inventory, days]
            done       (bool)    : Whether episode ended
        """
        inv      = int(np.clip(state[0],      0, self.max_inventory))
        days     = int(np.clip(state[1],      0, self.max_days))
        next_inv = int(np.clip(next_state[0], 0, self.max_inventory))
        next_day = int(np.clip(next_state[1], 0, self.max_days))

        current_q = self.q_table[inv, days, action]

        if done:
            target_q = reward
        else:
            best_next_q = np.max(self.q_table[next_inv, next_day])
            target_q    = reward + self.gamma * best_next_q

        # Bellman update
        self.q_table[inv, days, action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=10000):
        """
        Train agent over many episodes.

        Args:
            env          : AirlinePricingEnv instance
            num_episodes : Number of training episodes

        Returns:
            list: Total reward per episode
        """
        print("=" * 50)
        print("Q-Learning Training Started")
        print(f"  Episodes    : {num_episodes}")
        print(f"  Alpha       : {self.alpha}")
        print(f"  Gamma       : {self.gamma}")
        print(f"  Epsilon     : {self.epsilon} -> {self.epsilon_min}")
        print("=" * 50)

        for episode in range(num_episodes):
            obs, _        = env.reset()
            done          = False
            total_revenue = 0.0

            while not done:
                action                             = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done                               = terminated or truncated

                self.update_q_table(obs, action, reward, next_obs, done)

                obs            = next_obs
                total_revenue += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_revenue)

            # Print progress every 1000 episodes
            if (episode + 1) % 1000 == 0:
                avg = np.mean(self.episode_rewards[-1000:])
                print(f"  Episode {episode+1:>6}/{num_episodes} | "
                      f"Avg Revenue: Rs {avg:>10.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        print("\nTraining Complete!")
        print(f"Final Avg Revenue (last 1000 eps): "
              f"Rs {np.mean(self.episode_rewards[-1000:]):.2f}")

        return self.episode_rewards

    def evaluate(self, env, num_episodes=1000):
        """
        Evaluate trained agent with epsilon=0 (pure exploitation).

        Args:
            env          : AirlinePricingEnv instance
            num_episodes : Number of evaluation episodes

        Returns:
            dict: mean, std, min, max revenue
        """
        saved_epsilon = self.epsilon
        self.epsilon  = 0.0
        revenues      = []

        print(f"\nEvaluating over {num_episodes} episodes...")

        for episode in range(num_episodes):
            obs, _        = env.reset()
            done          = False
            total_revenue = 0.0

            while not done:
                action                             = self.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done                               = terminated or truncated
                total_revenue                     += reward

            revenues.append(total_revenue)

        self.epsilon = saved_epsilon

        results = {
            "mean_revenue" : round(np.mean(revenues), 2),
            "std_revenue"  : round(np.std(revenues), 2),
            "min_revenue"  : round(np.min(revenues), 2),
            "max_revenue"  : round(np.max(revenues), 2),
        }

        print("Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: Rs {v}")

        return results

    def plot_learning_curve(self):
        """Plot reward per episode over training."""
        import matplotlib.pyplot as plt

        rewards    = self.episode_rewards
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')

        plt.figure(figsize=(12, 5))
        plt.plot(rewards,    alpha=0.3, color='blue',  label='Per Episode')
        plt.plot(moving_avg, alpha=0.9, color='red',   label='100-Ep Moving Avg')
        plt.xlabel("Episode")
        plt.ylabel("Total Revenue (Rs)")
        plt.title("Q-Learning: Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("qlearning_learning_curve.png")
        plt.show()
        print("Plot saved as qlearning_learning_curve.png")

    def plot_policy(self):
        """Visualize learned policy as a heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get best action index for each (inventory, days) state
        policy = np.argmax(self.q_table, axis=2)

        plt.figure(figsize=(14, 7))
        sns.heatmap(
            policy,
            cmap       = "YlOrRd",
            xticklabels= 5,
            yticklabels= 5,
        )
        plt.xlabel("Days Remaining")
        plt.ylabel("Inventory Remaining")
        plt.title("Q-Learning: Learned Pricing Policy Heatmap\n"
                  "(Higher = More Expensive Price Selected)")
        plt.tight_layout()
        plt.savefig("qlearning_policy_heatmap.png")
        plt.show()
        print("Plot saved as qlearning_policy_heatmap.png")


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    env   = AirlinePricingEnv()
    agent = QLearningAgent()

    # Train
    agent.train(env, num_episodes=10000)

    # Evaluate
    results = agent.evaluate(env, num_episodes=1000)

    # Plot
    agent.plot_learning_curve()
    agent.plot_policy()
