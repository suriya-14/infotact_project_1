"""
    Heuristic agent that sets price based on remaining inventory and
    days left, without learning. Used as a baseline against DQN and
    Q-Learning in the dashboard's agent comparison sections.

    Logic:
        - Low inventory ratio (scarcity) -> push price up
        - Few days left with lots of inventory left -> push price down
          to clear stock before departure
        - Otherwise -> stay near a mid-range price
"""

import numpy as np

class InventoryBasedAgent:

    def __init__(
        self,
        prices: list,
        max_inventory: int,
        max_days: int,
        scarcity_weight: float = 0.6,
        urgency_weight: float = 0.4,
    ) -> None:
        self.prices = prices
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.scarcity_weight = scarcity_weight
        self.urgency_weight = urgency_weight
        self.num_prices = len(prices)

    def act(self, state) -> int:
        inventory_left, days_left = state[0], state[1]

        inventory_ratio = inventory_left / self.max_inventory
        days_ratio = days_left / self.max_days

        scarcity_score = 1.0 - inventory_ratio
        urgency_score = 1.0 - days_ratio

        clearance_pressure = urgency_score * inventory_ratio

        price_score = (
            self.scarcity_weight * scarcity_score
            - self.urgency_weight * clearance_pressure
        )
        price_score = float(np.clip(price_score, 0.0, 1.0))

        action = int(round(price_score * (self.num_prices - 1)))
        action = max(0, min(self.num_prices - 1, action))
        return action

    def get_price(self, state) -> float:
        return self.prices[self.act(state)]

    def update(self, state, action, reward, next_state, done) -> None:
        pass

    def reset(self) -> None:
        pass
