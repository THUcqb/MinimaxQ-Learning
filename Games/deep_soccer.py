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
        'video.frames_per_second': 10
    }

    move_vec = {
        0: [0,  0],
        1: [0, -1],
        2: [-1, 0],
        3: [0,  1],
        4: [1,  0],
    }

    def __init__(self, num_players=1, height=4, width=7):
        self.num_players = num_players
        self.height = height
        self.width = width
        self.MAX_STEPS_IN_ONE_EPISODE = 100
        # 1 stand, 4 direction, (N-1)teammates
        self.num_actions_for_one_player = 1 + 4 + num_players - 1
        self.num_actions_for_one_team = self.num_actions_for_one_player ** num_players
        self.action_space = spaces.Discrete(self.num_actions_for_one_team ** 2)
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
        self.state = np.zeros(
            (self.height, self.width, 2 * self.num_players + 1), dtype=bool)
        # 1 Put players
        player_locations = self.np_random.randint(
            0, self.height * (self.width // 2), 2 * self.num_players)
        for player in range(2 * self.num_players):
            if player < self.num_players:
                # team 0 on left half field
                self.state[(player + 1) * (self.height - 1) //
                           (self.num_players + 1), int(np.ceil(self.width / 4)), player] = 1
            else:
                # team 1 on right half field
                self.state[self.height - 1 - (player - self.num_players + 1) * (self.height - 1) //
                           (self.num_players + 1), self.width - 1 - int(np.ceil(self.width / 4)), player] = 1
        # 2 Give the ball to one player
        self.player_with_ball = self.np_random.choice(2 * self.num_players)
        self.state[:, :, -1] = self.state[:, :, self.player_with_ball]
        # 2 Put the ball in the middle
        # self.state[self.height // 2, self.width // 2, -1] = 1
        self.step_in_episode = 0
        return self.state

    def _step(self, action):
        # 1 Find the ball. Ball represented in state[:, :, -1]
        ball_x, ball_y = self._onehot_to_index(self.state[:, :, -1])

        # 2 Determine who the ball will go with
        #   Get all players standing at the ball's location
        if self.player_with_ball < self.num_players:
            opponents_at_ball = [player for player in range(self.num_players, 2 * self.num_players) if self.state[ball_x, ball_y, player]]
        else:
            opponents_at_ball = [player for player in range(self.num_players) if self.state[ball_x, ball_y, player]]
        #   The ball will go with this guy
        self.player_with_ball = self.np_random.choice(
            opponents_at_ball) if opponents_at_ball != [] else self.player_with_ball

        # 3 Step the players, for team 0 and team 1
        #   During this process, the ball's owner may change
        self.player_with_ball = self._step_team(0, action % self.num_actions_for_one_team, self.player_with_ball)
        self.player_with_ball = self._step_team(1, action // self.num_actions_for_one_team, self.player_with_ball)

        # 4 Step the ball
        if self.player_with_ball != None:
            self.state[:, :, -1] = self.state[:, :, self.player_with_ball]

        # 5 Judge, the goal have a height of 4 on the leftmost and rightmost side.
        reward = 0.0
        done = False
        new_ball_x, new_ball_y = self._onehot_to_index(self.state[:, :, -1])
        if new_ball_y == 0 and abs(new_ball_x - (self.height - 1) / 2) < 1:
            reward = -1.0
            done = True
        if new_ball_y == self.width - 1 and abs(new_ball_x - (self.height - 1) / 2) < 1:
            reward = 1.0
            done = True

        # Too many steps. Draw
        self.step_in_episode += 1
        if self.step_in_episode >= self.MAX_STEPS_IN_ONE_EPISODE:
            reward = 0.0
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
        rendered_rgb[:, :, 0] = np.sum(
            self.state[:, :, :self.num_players], axis=2)
        rendered_rgb[:, :, 1] = self.state[:, :, -1]
        rendered_rgb[:, :, 2] = np.sum(
            self.state[:, :, self.num_players:-1], axis=2)
        # show goal
        rendered_rgb[int(np.floor((self.height - 1) / 2)):1+int(np.ceil((self.height - 1) / 2)), 0, :] += 0.2
        rendered_rgb[int(np.floor((self.height - 1) / 2)):1+int(np.ceil((self.height - 1) / 2)), -1, :] += 0.2
        rendered_rgb /= np.max(rendered_rgb)
        # float to uint8
        rendered_rgb = (rendered_rgb * 255).round()
        # Amplifying the image
        rendered_rgb = np.kron(rendered_rgb, np.ones(
            (24, 24, 1))).astype(np.uint8)
        return rendered_rgb

    def _in_board(self, x, y):
        # Original setting hard coded
        return (0 <= x < self.height and 0 <= y < self.width)\
                and not (x == 0 and y == 0) and not (x == 0 and y == self.width-1)\
                and not (x == self.height-1 and y == 0) and not (x == self.height-1 and y == self.width-1)
    @staticmethod
    def _onehot_to_index(arr):
        '''Return the (x, y) indices of the one-hot 2d numpy array.'''
        return np.unravel_index(np.argmax(arr), arr.shape)

    MAX_PASSING_DISTANCE = 5

    def _trajectory_clear(self, holder, target):
        '''To see if people or ball on the line between the two player.

        Or the two players are too far away.'''
        x0, y0 = self._onehot_to_index(self.state[:, :, holder])
        x1, y1 = self._onehot_to_index(self.state[:, :, target])
        if (x0 - x1) ** 2 + (y0 - y1) ** 2 > DeepSoccer.MAX_PASSING_DISTANCE ** 2:
            return False
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
