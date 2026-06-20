import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim=2,
        action_dim=20,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=10000,
        target_update=10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        predicted_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path="models/dqn_weights.pth"):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path="models/dqn_weights.pth"):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from environment.airline_pricing_env import AirlinePricingEnv

    env = AirlinePricingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    episodes = 1000
    target_update_every = 10

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        if (episode + 1) % target_update_every == 0:
            agent.update_target()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} | Total Revenue: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_weights.pth")
    print("Training complete. Weights saved to models/dqn_weights.pth")
