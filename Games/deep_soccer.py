import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class DeepSoccer:
    # Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand]
    def __init__(self, height=40, width=50, pos_a_1=None, pos_a_2=None, pos_b_1=None, pos_b_2=None, ball_owner=0, draw_probability=0):
        # [3, 2]
        # [1, 1]
        if pos_a_1 is None:
            pos_a_1 = [np.random.randint(0, height), np.random.randint(0, int(width/2))]
        if pos_a_2 is None:
            while pos_a_2 is None or pos_a_1 == pos_a_2:
                pos_a_2 = [np.random.randint(0, height), np.random.randint(0, int(width/2))]
        if pos_b_1 is None:
            pos_b_1 = [np.random.randint(0, height), np.random.randint(int(width/2), width)]
        if pos_b_2 is None:
            while pos_b_2 is None or pos_b_1 == pos_b_2:
                pos_b_2 = [np.random.randint(0, height), np.random.randint(int(width/2), width)]
        self.agent_1 = [pos_a_1, 0]
        self.agent_2 = [pos_a_2, 1]
        self.agent_3 = [pos_b_1, 2]
        self.agent_4 = [pos_b_2, 3]
        self.agents = [self.agent_1, self.agent_2, self.agent_3, self.agent_4]
        self.height = height
        self.width = width
        self.positions = np.array([pos_a_1, pos_a_2, pos_b_1, pos_b_2])
        self.initPositions = np.array([pos_a_1, pos_a_2, pos_b_1, pos_b_2])
        self.ball_owner = ball_owner
        self.draw_probability = draw_probability
        self.step_count = 0
        self.terminal = False
        # init state

        self.soccer_grid = np.zeros([height, width, 5])
        for i, pos in enumerate(self.positions):
            # import pdb; pdb.set_trace()
            self.soccer_grid[pos[0], pos[1], i] = 1
        self.soccer_grid[self.agents[self.ball_owner][0][0],
                         self.agents[self.ball_owner][0][1], 4] = 1


    def reset(self, pos_a_1=None, pos_a_2=None, pos_b_1=None, pos_b_2=None, ball_owner=None):
        def update_position(pos, w_low, w_high, h_low=0, h_high=80):
            if pos is not None:
                return pos
            else:
                return [np.random.randint(h_low, h_high), np.random.randint(w_low, w_high)]
        for i, pos in enumerate([pos_a_1, pos_a_2, pos_b_1, pos_b_2]):
            if i <= 1:
                temp_w_low = 0
                temp_w = int(self.width/2)
            else:
                temp_w_low = int(self.width/2)
                temp_w = self.width
            self.initPositions[i] = update_position(pos, temp_w_low, temp_w)

        if ball_owner is None:
            ball_owner = self.choose_player()

        self.positions = self.initPositions.copy()
        self.ball_owner = ball_owner
        self.step_count = 0
        self.terminal = False
        # reset state
        self.soccer_grid = np.zeros([self.height, self.width, 5])
        for i, pos in [pos_a_1, pos_a_2, pos_b_1, pos_b_2]:
            self.soccer_grid[pos[0], pos[1], i] = 1
        self.soccer_grid[self.agents[self.ball_owner][0][0], self.agents[self.ball_owner][0][0], 4] = 1
        return self.soccer_grid

    def step(self, action_a, action_b):

        status = 0
        no_player = -1
        self.step_count += 1
        if self.step_count >= 2000:     # run too many steps? give it a draw
            self.terminal = True
            status = -2
            return self.soccer_grid, status, no_player, self.terminal
        first = self.choose_side()
        second = 1 - first
        action_in_play = [action_a, action_b]
        status = self.move(first, action_in_play[first])    # init win? return goal, init, terminal
        self.update_state()
        if status == 1:
            self.terminal = True
            return self.soccer_grid, status, first, self.terminal

        status = self.move(second, action_in_play[1 - first])  # second win? return goal, second, terminal
        self.update_state()
        if status == 1:
            self.terminal = True
            return self.soccer_grid, status, second, self.terminal
        return self.soccer_grid, 0, no_player, self.terminal    # the game is going on.

    def move(self, player, action_move):
        opponent = 1 - player
        player_move = self.action_to_move(action_move)
        if player_move[0] == 'passball':
            self.ball_owner = player*2+1
        elif player_move[1] == 'passball':
            self.ball_owner = player*2
        else:
            new_position = [self.positions[player*2] + player_move[0],
                            self.positions[player*2+1] + player_move[1]]  # new position for the current player

        # dodging
        if (new_position[0] == self.positions[opponent*2]).all() and self.ball_owner == player*2:
            self.ball_owner = self.agents[opponent * 2][1]
        if (new_position[0] == self.positions[opponent*2+1]).all() and self.ball_owner == player*2:
            self.ball_owner = self.agents[opponent*2+1][1]
        if (new_position[1] == self.positions[opponent*2]).all() and self.ball_owner == player*2+1:
            self.ball_owner = self.agents[opponent * 2][1]
        if (new_position[1] == self.positions[opponent*2+1]).all() and self.ball_owner == player*2+1:
            self.ball_owner = self.agents[opponent*2+1][1]


        # goal
        if self.ball_owner == player*2 and self.is_ingoal(new_position[0][0], new_position[0][1], player):
            return 1
        elif self.ball_owner == player*2+1 and self.is_ingoal(new_position[1][0], new_position[1][1], player):
            return 1

        # on board
        if self.is_inboard(*new_position[0]) and \
                not self.conflict(my_pos=new_position[0], friend_pos=new_position[1], all_pos=self.positions, opponent=opponent):
            self.positions[player*2] = new_position[0]
        if self.is_inboard(*new_position[1]) and \
                not self.conflict(my_pos=new_position[0], friend_pos=new_position[1], all_pos=self.positions, opponent=opponent):
            self.positions[player*2+1] = new_position[1]

        # invalid action -> nothing happens. return 0 means no goal happens.
        return 0

    def update_state(self):
        # update state
        self.soccer_grid[:] = 0
        for i, pos in self.positions:
            self.soccer_grid[pos[0], pos[1], i] = 1
        self.soccer_grid[self.agents[self.ball_owner][0][0],
                         self.agents[self.ball_owner][0][1], 4] = 1
        
    @staticmethod
    def action_to_move(action):
        # Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand, 5:pass the ball]
        switcher = {
            0: [0, -1],
            1: [-1, 0],
            2: [0,  1],
            3: [1,  0],
            4: [0,  0],
            5: 'passball'
        }
        return [switcher.get(action[0]), switcher.get(action[1])]

    def is_ingoal(self, h, w, player):
        assert(player == 0 or player == 1)
        h_gate = 6
        if player == 0:
            if int(self.height/2)+int(h_gate/2) >= h >= int(self.height/2)-int(h_gate/2) and w == self.width:
                return True
        elif player == 1:
            if int(self.height/2)+int(h_gate/2) >= h >= int(self.height/2)-int(h_gate/2) and w == -1:
                return True
        return False

    def is_inboard(self, h, w):
        return 0 <= h < self.height and 0 <= w < self.width

    def conflict(self, my_pos, friend_pos, all_pos, opponent):
        if (my_pos != friend_pos).any() and \
           (my_pos != all_pos[2*opponent]).any() and \
           (my_pos != all_pos[2*opponent+1]).any():
            return True
        else:
            return False

    @staticmethod
    def choose_player():
        return np.random.randint(0, 4)

    @staticmethod
    def choose_side():
        return np.random.randint(0, 2)

    def render(self, num):
        render_grid = np.sum(self.soccer_grid[:, :, 0:-1], axis=2)*2 + 5 * self.soccer_grid[:, :, -1]
        #import pdb; pdb.set_trace()
        render_grid = render_grid.reshape(self.height, self.width, 1) * 0.13
        matplotlib.image.imsave(str(num)+'.png',  np.tile(render_grid, (1, 1, 3)))


if __name__ == '__main__':
    s = DeepSoccer()
    s.render(-1)

    actions = np.array([[[2, 2], [2, 2]], [[2, 2], [2, 2]], [[2, 2], [2, 2]], [[2, 2], [2, 2]], [[2, 2], [2, 2]]])
    # actions = [[0, 4], [0, 4], [0, 4], [1, 4], [0, 4]]
    for i, action in enumerate(actions):
        s.step(*action)
        print(action)
        s.render(i)
