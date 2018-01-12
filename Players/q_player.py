import numpy as np


class QPlayer:

    def __init__(self, num_states, num_actions, decay, eps, gamma):
        self.decay = decay
        self.eps = eps
        self.gamma = gamma
        self.alpha = 1
        self.V = np.ones(num_states)
        self.Q = np.ones((num_states, num_actions))
        self.pi = np.ones((num_states, num_actions)) / num_actions
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning = True

    def choose_action(self, state):
        if self.learning and np.random.rand() < self.eps:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def bellman_update(self, state, next_state, actions, reward):
        if not self.learning:
            return
        action_a, action_b = actions
        self.Q[state, action_a] = (1 - self.alpha) * self.Q[state, action_a] + \
            self.alpha * (reward + self.gamma * self.V[next_state])
        best_action = np.argmax(self.Q[state])
        self.pi[state] = np.zeros(self.num_actions)
        self.pi[state, best_action] = 1
        self.V[state] = self.Q[state, best_action]
        self.alpha *= self.decay

    def policy_for_state(self, state):
        for i in range(self.num_actions):
            print("Actions %d : %f" % (i, self.pi[state, i]))


if __name__ == '__main__':
    print('THIS IS A Q LEARNING CLASS')
