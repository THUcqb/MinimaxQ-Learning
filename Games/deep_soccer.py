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

    move_vec = {
        0: [0,  0],
        1: [0, -1],
        2: [-1, 0],
        3: [0,  1],
        4: [1,  0],
    }

    def __init__(self, num_players=2, height=8, width=10):
        self.num_players = num_players
        self.height = height
        self.width = width
        # 1 stand, 4 direction, (N-1)teammates
        self.num_actions_for_one_player = 1 + 4 + self.num_players - 1
        self.action_space = spaces.Discrete(self.num_actions_for_one_player ** num_players)
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
        # 1 Put players
        player_locations = self.np_random.randint(0, self.height * (self.width // 2), 2 * self.num_players)
        for player, loc in enumerate(player_locations):
            if player < self.num_players:
                # team 0 randomly on left half field
                self.state[loc % self.height, loc // self.height, player] = 1
            else:
                # team 1 randomly on right half field
                self.state[loc % self.height, self.width - 1 - loc // self.height, player] = 1
        # 2 Put the ball
        ball_owner = self.np_random.randint(2 * self.num_players)
        self.state[:, :, -1] = self.state[:, :, ball_owner]
        return self.state

    def _step(self, action):
        # 1 Find the ball. Ball represented in state[:, :, -1]
        ball_x, ball_y = self._onehot_to_index(self.state[:, :, -1])

        # 2 Determine who the ball will go with
        #   Get all players standing at the ball's location
        players_at_ball = [player for player in range(2 * self.num_players)
                                  if self.state[ball_x, ball_y, player]]
        #   The ball will go with this guy
        player_with_ball = self.np_random.choice(players_at_ball)

        # 3 Step the players, for team 0 and team 1
        #   During this process, the ball's owner may change
        #   Take random action for team 1 in Q-learning.
        player_with_ball = self._step_team(0, action, player_with_ball)
        random_action = self.np_random.choice(self.action_space.n)
        player_with_ball = self._step_team(1, random_action, player_with_ball)

        # 4 Step the ball
        self.state[:, :, -1] = self.state[:, :, player_with_ball]

        # 5 Judge, the goal have a height of 4 on the leftest and rightest side.
        reward = 0.0
        done = False
        new_ball_x, new_ball_y = self._onehot_to_index(self.state[:, :, -1])
        if new_ball_y == 0 and abs(new_ball_x - self.height / 2) < 2:
            reward = -1.0
            done = True
        if new_ball_y == self.width-1 and abs(new_ball_x - self.height / 2) < 3:
            reward = 1.0
            done = True

        return self.state, reward, done, {}

    def _step_team(self, team, action, player_with_ball):
        new_player_with_ball = None
        for player in range(team * self.num_players, (team + 1) * self.num_players):
            action_t = action % self.num_actions_for_one_player
            action //= self.num_actions_for_one_player
            if action_t < 5:
                # Stand / move
                x, y = self._onehot_to_index(self.state[:, :, player])
                if self._in_board(x + self.move_vec[action_t][0], y + self.move_vec[action_t][1]):
                    self.state[x, y, player] = 0
                    x += self.move_vec[action_t][0]
                    y += self.move_vec[action_t][1]
                    self.state[x, y, player] = 1
            else:
                # Pass the ball only if you have the ball
                if player_with_ball != player:
                    continue

                # To whom you want to pass
                target = action_t - 5 + team * self.num_players

                # action doesn't include passing to oneself
                # so fix the misalignment
                if target >= player:
                    target += 1
                # now target is from 0~N-1 for team0, N~2N-1 for team1,
                # && target != player

                if self._trajectory_clear(player, target):
                    new_player_with_ball = target
        if new_player_with_ball is not None:
            return new_player_with_ball
        else:
            return player_with_ball

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
        return rendered_rgb

    def _in_board(self, x, y):
        return 0 <= x < self.height and 0 <= y < self.width

    @staticmethod
    def _onehot_to_index(arr):
        '''Return the (x, y) indices of the one-hot 2d numpy array.'''
        return np.unravel_index(np.argmax(arr), arr.shape)

    def _trajectory_clear(self, holder, target):
        '''To see if people or ball on the line between the two player.'''
        x0, y0 = self._onehot_to_index(self.state[:, :, holder])
        x1, y1 = self._onehot_to_index(self.state[:, :, target])
        trajectory = self._trajectory(x0, y0, x1, y1)
        # Take people and ball as obstacles
        obstacles = np.sum(self.state, axis=-1)

        for x, y in trajectory:
            if obstacles[x, y]:
                return False
        return True

    @staticmethod
    def _trajectory(x0, y0, x1, y1):
        '''Wrapper for compute trajectory between two points.'''
        rearranged_xy = False
        if abs(y0 - y1) > abs(x0 - x1):
            x0, y0, x1, y1 = y0, x0, y1, x1
            rearranged_xy = True
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        trajectory = DeepSoccer._trajectory_Bresenham(x0, y0, x1, y1)
        if rearranged_xy:
            trajectory = [(y, x) for (x, y) in trajectory]
        return trajectory

    @staticmethod
    def _trajectory_Bresenham(x0, y0, x1, y1):
        '''Return the trajectory to pass the ball between two people.

        Refer to [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)

        Params should satisfy: abs(x1-x0) >= abs(y1-y0), x1 > x0
        '''
        trajectory = []
        dx, dy = x1 - x0, y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        y = y0
        D = 2 * dy - dx

        for x in range(x0, x1):
            trajectory.append((x, y))
            if D > 0:
                y = y + yi
                D -= 2 * dx
            D += 2 * dy

        return trajectory[1:]


if __name__ == '__main__':
    s = DeepSoccer()
    print(s.reset())
    matplotlib.image.imsave('cqb.png', s.render())
