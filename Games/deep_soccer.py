import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class DeepSoccer(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, num_players=3, height=8, width=10):
        self.num_players = num_players
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(5**num_players)
        self.observation_space = spaces.Box(
            np.zeros((height, width, 2 * num_players + 1), dtype=bool),
            np.ones((height, width, 2 * num_players + 1), dtype=bool),
        )
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        '''Randomly put the players on the grid and give the ball to someone.

        State of shape H*W*(2 * number of players + 1(ball))
        '''
        self.state = np.zeros((self.height, self.width, 2 * self.num_players + 1), dtype=bool)
        # put players
        player_locations = self.np_random.randint(0, self.height * self.width, 2 * self.num_players)
        for player, loc in enumerate(player_locations):
            self.state[loc // self.width, loc % self.width, player] = 1
        # put the ball
        ball_owner = self.np_random.randint(2 * self.num_players)
        self.state[player_locations[ball_owner] // self.width, player_locations[ball_owner] % self.width, 2 * self.num_players] = 1
        return self.state

    def _step(self, action):
        # TODO
        reward = 1.0
        done = False
        return self.state, reward, done, {}

    def _render(self, mode='human', close=False):
        '''Return rgb array (height, width, 3)

        Red for team A
        Green for the ball
        Blue for team B
        '''
        rendered_rgb = np.zeros([self.height, self.width, 3])
        rendered_rgb[:, :, 0]= np.sum(self.state[:, :, :self.num_players], axis=2)
        rendered_rgb[:, :, 1]= self.state[:, :, -1]
        rendered_rgb[:, :, 2]= np.sum(self.state[:, :, self.num_players:-1], axis=2)
        rendered_rgb /= np.max(rendered_rgb)
        # float to uint8
        rendered_rgb = (rendered_rgb * 255).round()
        # Amplifying the image
        rendered_rgb = np.kron(rendered_rgb, np.ones((16, 16, 1))).astype(np.uint8)

        # DEBUG save and view the image
        matplotlib.image.imsave('cqb.png', rendered_rgb)
        return rendered_rgb

    move_vec = {
        0: [0, -1],
        1: [-1, 0],
        2: [0,  1],
        3: [1,  0],
        4: [0,  0],
    }

    def _in_goal(self, h, w, player):
        # TODO
        # assert(player == 0 or player == 1)
        # if player == 0:
        #     if (h == 1 or h == 2) and w == 5:
        #         return True
        # elif player == 1:
        #     if (h == 1 or h == 2) and w == -1:
        #         return True
        # return False
        pass

    def _in_board(self, h, w):
        return 0 <= h < self.height and 0 <= w < self.width


if __name__ == '__main__':
    s = DeepSoccer()
    print(s.reset())
    s.render()
