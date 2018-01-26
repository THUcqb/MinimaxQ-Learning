import numpy as np
from scipy.optimize import linprog


class MinimaxQPlayer:

    def __init__(self, num_states, num_actions_a, num_actions_b, decay, eps, gamma):
        self.decay = decay  # the default value for decay is 0.9 according to the paper
        self.eps = eps    # eps is the exploration
        self.gamma = gamma  # gamma the dicounted factor
        self.alpha = 1     #  update steps
        self.V = np.ones(num_states)   # value function
        self.Q = np.ones((num_states, num_actions_a, num_actions_b))  # minimax Q function
        self.pi = np.ones((num_states, num_actions_a)) / num_actions_a  # player 1's policy
        self.num_states = num_states  #TODO: check soccer to state 
        self.num_actions_a = num_actions_a  # 5 for soccer
        self.num_actions_b = num_actions_b  # 5 for soccer
        self.learning = True  # is this in the learning stage

    def choose_action(self, state):
        if self.learning and np.random.rand() < self.eps:  # e-greedy
            action = np.random.randint(self.num_actions_a)
        else:
            action = self.weighted_action_choice(state, self.num_actions_a)  # check actions
        return action

    def weighted_action_choice(self, state, num_action):
        action = np.random.choice(num_action, p=np.squeeze(self.pi[state]))
        return action

    def bellman_update(self, state, next_state, actions, reward):
        if not self.learning:  # return without any reward. What does bellman_update mean?
            return
        action_a, action_b = actions
        self.Q[state, action_a, action_b] = (1 - self.alpha) * self.Q[state, action_a, action_b] + \
            self.alpha * (reward + self.gamma * self.V[next_state])
        # harry: this is not elegant. better update V and pi in two functions
        self.V[state] = self.update_policy(state)  # EQUIVALENT TO : min(np.sum(self.Q[state].T * self.pi[state], axis=1))
        self.alpha *= self.decay   # this can be something to tune

    def update_policy(self, state, retry=False):
        #  x is the policy we want to optimize.
        c = np.zeros(self.num_actions_a + 1)
        c[0] = -1
        A_ub = np.ones((self.num_actions_b, self.num_actions_a + 1))
        A_ub[:, 1:] = -self.Q[state].T
        b_ub = np.zeros(self.num_actions_b)
        A_eq = np.ones((1, self.num_actions_a + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.num_actions_a

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if res.success:
            self.pi[state] = res.x[1:]
        elif not retry:
            return self.update_policy(state, retry=True)
        else:
            print("Alert : %s" % res.message)
            return self.V[state]

        return res.x[0]

    def policy_for_state(self, state):
        for i in range(self.num_actions_a):
            print("Actions %d : %f" % (i, self.pi[state, i]))


if __name__ == '__main__':

    def testupdate_policy():
        m = MinimaxQPlayer(1, 2, 2, 1e-4, 0.2, 0.9)
        m.Q[0] = [[0, 1], [1, 0.5]]
        print(m.pi)
        m.update_policy(0)
        print(m.pi)

    testupdate_policy()
