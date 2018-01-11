import numpy as np


class Soccer:

    # Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand]

    def __init__(self, height=4, width=5, pos_a=None, pos_b=None, ball_owner=0, draw_probability=0):
        # [3, 2]
        # [1, 1]
        if pos_a is None:
            pos_a = [2, 1]
        if pos_b is None:
            pos_b = [1, 3]

        self.height = height
        self.width = width
        self.positions = np.array([pos_a, pos_b])
        self.initPositions = np.array([pos_a, pos_b])
        self.ball_owner = ball_owner
        self.draw_probability = draw_probability
        self.step_count = 0
        self.terminal = False

    def reset(self, pos_a=None, pos_b=None, ball_owner=None):
        if pos_a is not None:
            self.initPositions[0] = pos_a
        else:
            self.initPositions[0] = [np.random.randint(self.height), 0]

        if pos_b is not None:
            self.initPositions[1] = pos_b
        else:
            self.initPositions[1] = [np.random.randint(self.height), self.width-1]

        if ball_owner is None:
            ball_owner = self.choose_player()

        self.positions = self.initPositions.copy()
        self.ball_owner = ball_owner
        self.step_count = 0
        self.terminal = False

    def step(self, action_a, action_b):
        status = np.NaN
        if self.step_count >= 10:
            self.terminal = True
            return status, self.terminal
        first = self.choose_player()
        second = 1 - first
        action_in_play = [action_a, action_b]
        m1 = self.move(first, action_in_play[first])
        if m1 == 1:
            print("Goal!! Need to reset.")
            return m1, first
        return self.move(second, action_in_play[1 - first]), second

    def move(self, player, action_move):
        opponent = 1 - player
        new_position = self.positions[player] + self.action_to_move(action_move)

        # If it's opponent position
        if (new_position == self.positions[opponent]).all():
            self.ball_owner = opponent
        # If it's the goal
        elif self.ball_owner is player and self.is_ingoal(new_position[0], new_position[1], player) :
            return 1
        # If it's in board
        elif self.is_inboard(*new_position):  # * here is to use component of new_position as argument
            self.positions[player] = new_position
        # invalid action -> nothing happens. return 0 means no goal happens.
        return 0

    def action_to_move(self, action):
        # Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand]
        switcher = {
            0: [0, -1],
            1: [-1, 0],
            2: [0,  1],
            3: [1,  0],
            4: [0,  0],
        }
        return switcher.get(action)

    def is_ingoal(self, h, w, player):
        assert(player == 0 or player == 1)
        if player == 0:
            if (h == 1 or h == 2) and w == 5:
                return True
        elif player == 1:
            if (h == 1 or h == 2) and w == -1:
                return True
        return False

    def is_inboard(self, h, w):
        return 0 <= h < self.height and 0 <= w < self.width

    def choose_player(self):
        return np.random.randint(0, 2)

    def render(self, positions=None, ball_owner=None):
        positions = self.positions if positions is None else np.array(positions)
        ball_owner = self.ball_owner if ball_owner is None else ball_owner
        #import pdb; pdb.set_trace()
        board = ''
        for h in range(self.height):
            for w in range(self.width):
                if ([h, w] == positions[0]).all():
                    board += 'A' if ball_owner is 0 else 'a'
                elif ([h, w] == positions[1]).all():
                    board += 'B' if ball_owner is 1 else 'b'
                else:
                    board += '-'
            board += '\n'

        print(board)


if __name__ == '__main__':
    s = Soccer()
    s.render()
    actions = [[2, 1], [2, 4], [2, 3], [2, 0], [2, 0]]
    # actions = [[0, 4], [0, 4], [0, 4], [1, 4], [0, 4]]
    for action in actions:
        s.step(*action)
        print(action)
        s.render()
