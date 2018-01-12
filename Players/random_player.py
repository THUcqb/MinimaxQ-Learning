import numpy as np


class RandomPlayer:

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def choose_action(self, state, max_action=None):
        max_action = self.num_actions if max_action is None else max_action
        return np.random.randint(max_action)

    def bellman_update(self, state, next_state, actions, reward, max_actions=None):
        pass
