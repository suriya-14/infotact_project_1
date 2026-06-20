# =============================================================================
# qlearning_agent.py
# Author      : Member 2
# Description : Tabular Q-Learning agent for airline dynamic pricing.
#               Uses bucketed state space for faster convergence.
# =============================================================================

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.airline_pricing_env import AirlinePricingEnv


class QLearningAgent:
    """
    Tabular Q-Learning agent with bucketed state space.

    KEY IMPROVEMENT over naive Q-table:
    Instead of tracking all 51 inventory levels and 31 day levels
    separately (51 x 31 x 20 = 31,620 values to learn), we bucket them
    into 10 groups each (10 x 10 x 20 = 2,000 values).

    This means the agent converges ~15x faster.

    Buckets:
        Inventory: [0-4, 5-9, 10-14, ..., 45-50] → 10 buckets
        Days     : [0-2, 3-5, 6-8,  ..., 28-30] → 10 buckets

    Bellman Update:
        Q(S,A) <- Q(S,A) + alpha * [R + gamma * max(Q(S',A')) - Q(S,A)]
    """

    def __init__(
        self,
        max_inventory      = 50,
        max_days           = 30,
        num_actions        = 20,
        n_inventory_buckets= 10,
        n_day_buckets      = 10,
        alpha              = 0.2,
        gamma              = 0.99,
        epsilon            = 1.0,
        epsilon_min        = 0.01,
        epsilon_decay      = 0.9995,
    ):
        self.max_inventory       = max_inventory
        self.max_days            = max_days
        self.num_actions         = num_actions
        self.n_inventory_buckets = n_inventory_buckets
        self.n_day_buckets       = n_day_buckets
        self.alpha               = alpha
        self.gamma               = gamma
        self.epsilon             = epsilon
        self.epsilon_min         = epsilon_min
        self.epsilon_decay       = epsilon_decay

        # OPTIMISTIC initialization — start Q-values HIGH (200,000)
        # This encourages the agent to try all actions at least once
        # because any unexplored action "looks" better than a known bad one
        self.q_table = np.full(
            (n_inventory_buckets, n_day_buckets, num_actions),
            fill_value=200_000.0
        )

        self.episode_rewards = []

    def _get_bucket(self, state):
        """
        Convert raw state [inventory, days] into bucket indices.

        Example (max_inventory=50, n_buckets=10):
            inventory=47 → bucket 9
            inventory=23 → bucket 4
            days=28      → bucket 9
            days=5       → bucket 1
        """
        inventory = float(state[0])
        days      = float(state[1])

        inv_bucket = int(np.clip(
            inventory / self.max_inventory * self.n_inventory_buckets,
            0, self.n_inventory_buckets - 1
        ))
        day_bucket = int(np.clip(
            days / self.max_days * self.n_day_buckets,
            0, self.n_day_buckets - 1
        ))

        return inv_bucket, day_bucket

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state (np.array): [inventory, days_remaining] from env

        Returns:
            int: price action index (0-19)
        """
        inv_b, day_b = self._get_bucket(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)   # EXPLORE
        else:
            return int(np.argmax(self.q_table[inv_b, day_b]))  # EXPLOIT

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
        inv_b,      day_b      = self._get_bucket(state)
        next_inv_b, next_day_b = self._get_bucket(next_state)

        current_q = self.q_table[inv_b, day_b, action]

        if done:
            target_q = reward
        else:
            best_next_q = np.max(self.q_table[next_inv_b, next_day_b])
            target_q    = reward + self.gamma * best_next_q

        # Bellman update
        self.q_table[inv_b, day_b, action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=20000):
        """
        Train agent over many episodes.

        Args:
            env          : AirlinePricingEnv instance
            num_episodes : Number of training episodes

        Returns:
            list: Total reward per episode
        """
        print("=" * 55)
        print("Q-Learning Training Started")
        print(f"  Episodes         : {num_episodes}")
        print(f"  Alpha            : {self.alpha}")
        print(f"  Gamma            : {self.gamma}")
        print(f"  Epsilon decay    : {self.epsilon_decay}")
        print(f"  Q-table shape    : {self.q_table.shape}")
        print(f"  Q-table init val : 200,000 (optimistic)")
        print("=" * 55)

        for episode in range(num_episodes):
            obs, _        = env.reset()
            done          = False
            total_revenue = 0.0

            while not done:
                action                                     = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done                                       = terminated or truncated

                self.update_q_table(obs, action, reward, next_obs, done)

                obs            = next_obs
                total_revenue += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_revenue)

            if (episode + 1) % 2000 == 0:
                avg = np.mean(self.episode_rewards[-2000:])
                print(f"  Episode {episode+1:>6}/{num_episodes} | "
                      f"Avg Revenue: Rs {avg:>10.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        print("\nTraining Complete!")
        print(f"Final Avg Revenue (last 2000 eps): "
              f"Rs {np.mean(self.episode_rewards[-2000:]):.2f}")

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

        for _ in range(num_episodes):
            obs, _        = env.reset()
            done          = False
            total_revenue = 0.0

            while not done:
                action                                     = self.select_action(obs)
                obs, reward, terminated, truncated, _      = env.step(action)
                done                                       = terminated or truncated
                total_revenue                             += reward

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
            print(f"  {k}: Rs {v:,.2f}")

        return results

    def plot_learning_curve(self):
        """Plot reward per episode over training."""
        import matplotlib.pyplot as plt

        rewards    = self.episode_rewards
        window     = min(500, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(12, 5))
        plt.plot(rewards,    alpha=0.2, color='blue', label='Per Episode')
        plt.plot(moving_avg, alpha=0.9, color='red',  label=f'{window}-Ep Moving Avg')
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

        policy       = np.argmax(self.q_table, axis=2)
        prices_menu  = list(range(500, 10001, 500))
        price_policy = np.vectorize(lambda x: prices_menu[x])(policy)

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            price_policy,
            cmap        = "YlOrRd",
            annot       = True,
            fmt         = ".0f",
            xticklabels = [f"{int(i/10*30)}d" for i in range(10)],
            yticklabels = [f"{int(i/10*50)}s" for i in range(10)],
        )
        plt.xlabel("Days Remaining (bucketed)")
        plt.ylabel("Inventory Remaining (bucketed)")
        plt.title("Q-Learning: Learned Pricing Policy (Rs)\n"
                  "Higher value = Higher price selected")
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

    agent.train(env, num_episodes=20000)
    agent.evaluate(env, num_episodes=1000)
    agent.plot_learning_curve()
    agent.plot_policy()
