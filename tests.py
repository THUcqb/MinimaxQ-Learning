import numpy as np
import matplotlib.pyplot as plt
from Players import *
from Games import *


class Tester:

    def __init__(self, game, playerA=None, playerB=None):
        self.game = game
        self.playerA = playerA
        self.playerB = playerB

    def result_to_reward(self, result, action_a=None, action_b=None):
        return result

    def plot_policy(self, player):
        for state in range(player.num_states):
            print("\n=================")
            self.game.render(*self.state_to_board(state))
            # print("State value: %s" % player.V[state])
            print(player.Q[state])
            player.policy_for_state(state)

    @staticmethod
    def plot_result(wins):
        len_wins = len(wins)
        sum_wins = (wins == [[0], [1], [-1]]).sum(1)
        print("Wins A : %d (%0.1f%%)" % (sum_wins[0], (100. * sum_wins[0] / len_wins)))
        print("Wins B : %d (%0.1f%%)" % (sum_wins[1], (100. * sum_wins[1] / len_wins)))
        print("Draws  : %d (%0.1f%%)" % (sum_wins[2], (100. * sum_wins[2] / len_wins)))

        plt.plot((wins == 0).cumsum())
        plt.plot((wins == 1).cumsum())
        plt.legend(('Winstate_a', 'Winstate_b'), loc=(0.6, 0.8))
        plt.show()


class SoccerTester(Tester):

    def __init__(self, game):
        Tester.__init__(self, game)

    def board_to_state(self):
        game = self.game
        height_a, width_a = game.positions[0]
        height_b, width_b = game.positions[1]
        state_a = width_a * game.height + height_a
        state_b = width_b * game.height + height_b
        state = (state_a * (game.width * game.height - 1) + state_b) + (game.width * game.height) * (game.width * game.height - 1) * game.ball_owner
        return state

    def state_to_board(self, state):
        game = self.game
        ball_owner = state // ((game.width * game.height) * (game.width * game.height - 1))
        state = state % ((game.width * game.height) * (game.width * game.height - 1))

        state_a = state // (game.width * game.height - 1)
        state_b = state % (game.width * game.height - 1)

        height_a = state_a % game.height
        width_a = state_a // game.height
        height_b = state_b % game.height
        width_b = state_b // game.height

        return [[[height_a, width_a], [height_b, width_b]], ball_owner]

    def result_to_reward(self, result, action_a=None, action_b=None):
        factor = 1
        return Tester.result_to_reward(self, result) * factor


def testGame(playerA, playerB, gameTester, iterations):
    wins = np.zeros(iterations)

    for i in np.arange(iterations):
        if i % (iterations / 10) == 0:
            print("%d%%" % (i * 100 / iterations))
        gameTester.game.reset()

        while not gameTester.game.terminal:
            state = gameTester.board_to_state()
            action_a = playerA.choose_action(state)
            action_b = playerB.choose_action(state)
            result, player_ind,terminal = gameTester.game.step(action_a, action_b)
            reward = gameTester.result_to_reward(result, action_a, action_b)
            new_state = gameTester.board_to_state()
            if terminal is True:
                if player_ind == 0:
                    playerA.bellman_update(state, new_state, [action_a, action_b], reward)
                    playerB.bellman_update(state, new_state, [action_b, action_a], -reward)
                elif player_ind == 1:
                    playerA.bellman_update(state, new_state, [action_a, action_b], -reward)
                    playerB.bellman_update(state, new_state, [action_b, action_a], reward)
                elif player_ind == -1:
                    print("The game is draw!")
            playerA.bellman_update(state, new_state, [action_a, action_b], 0)
            playerB.bellman_update(state, new_state, [action_b, action_a], 0)

        wins[i] = player_ind
    return wins


def testSoccer(iterations):
    board_height = 4
    board_width = 5
    num_states = (board_width * board_height) * (board_width * board_height - 1) * 2
    num_actions = 5
    draw_probability = 0.01
    decay = 10**(-2. / iterations * 0.05)

    ### CHOOSE PLAYER_A TYPE
    # playerA = RandomPlayer(num_actions)
    playerA = MinimaxQPlayer(num_states, num_actions, num_actions, decay=decay, eps=0.3, gamma=1-draw_probability)
    # playerA = QPlayer(num_states, num_actions, decay=decay, eps=0.2, gamma=1-draw_probability)
    # playerA = np.load('state_avedPlayersminimax/Q_SoccerA_100000.npy').item()

    ### CHOOSE PLAYER_B TYPE
    playerB = RandomPlayer(num_actions)
    # playerB = MinimaxQPlayer(num_states, num_actions, num_actions, decay=decay, eps=0.2, gamma=1-draw_probability)
    # playerB = QPlayer(num_states, num_actions, decay=decay, eps=0.2, gamma=1-draw_probability)
    # playerB = np.load('state_avedPlayers/Q_SoccerB_100000.npy').item()

    ### INSTANTIATE GAME AND TESTER
    game = Soccer(board_height, board_width, draw_probability=draw_probability)
    tester = SoccerTester(game)

    ### RUN TEST
    wins = testGame(playerA, playerB, tester, iterations)

    ### DISPLAY RESULTS
    tester.plot_policy(playerA)
    # tester.plot_policy(playerB)
    tester.plot_result(wins)

    # np.state_ave("SoccerA_10000", playerA)
    # np.state_ave("SoccerB_10000", playerB)


def testSoccerPerformance():
    board_height, board_width = 4, 5
    num_states = (board_width * board_height) * (board_width * board_height - 1) * 2
    num_actions = 5
    draw_probability = 0.01

    ### INSTANTIATE GAME
    game = Soccer(height=board_height, width=board_width, draw_probability=draw_probability)

    print("AIM : EVALUATE OUR MINIMAX Q 'PLAYER A' TRAINED OVER 100.000 ITERATIONS")
    print("METHOD : MAKE IT FIGHT AGAINST A DETERMINISTIC PLAYER\n \
        AGAINST THERE EXIST A DETERMINISTIC POLICY THAT ALWAYS WINS")

    print("\n=======================================================")
    print("STEP 1: CREATE A DETERMINISTIC 'PLAYER B' TO FIGHT WITH")

    # CHOOSE PLAYER_B AS Q LEARNER
    playerB = QPlayer(num_states, num_actions, decay=1-1e-4, eps=0.3, gamma=1-draw_probability)

    ### TRAIN IT AGAINST ANOTHER Q LEARNER
    print("\n1.1 - TRAIN OUR 'PLAYER B' (Q LEARNER) AGAINST ANOTHER Q LEARNER - 5000 games")
    playerA1 = QPlayer(num_states, num_actions, decay=1-1e-4, eps=0.5, gamma=1-draw_probability)
    tester = SoccerTester(game)
    wins = testGame(playerA1, playerB, tester, 5000)

    ### TRAIN A Q LEARNER TO BEAT IT
    print("\n1.2 - TRAIN ANOTHER Q LEARNER TO BEAT 'PLAYER B' - 10000 games")
    print("('PLAYER B' is not learning anymore)")
    playerB.learning = False
    playerA2 = QPlayer(num_states, num_actions, decay=1-1e-4, eps=0.3, gamma=1-draw_probability)
    wins = testGame(playerA2, playerB, tester, 10000)
    tester.plot_result(wins)

    ### CHECK THIS Q LEARNER
    print("\n1.3 - CHECK THIS Q LEARNER ALWAYS BEAT 'PLAYER B' - 1000 games")
    print("(This step is facultative -- 'PLAYER B' should be always losing)")
    print("(Note: If it is not the case, relaunch program)")
    playerA2.learning = False
    wins = testGame(playerA2, playerB, tester, 1000)
    tester.plot_result(wins)

    ### MAKE FIGHT ! PLAYER A vs PLAYER B
    print("\n\n======================================================")
    print("STEP 2: MAKE PLAYER 'A' FIGHT 'PLAYER B' - 10000 games")
    playerA3 = np.load('state_avedPlayers/minimaxQ_SoccerA_100000.npy').item()
    playerA3.learning = False
    wins = testGame(playerA3, playerB, tester, 10000)
    tester.plot_result(wins)

    v = playerA2.pi == 1
    prod = sum(playerA2.pi[v] * playerA3.pi[v])
    print('\nApproximate percentage of correct actions : %0.1f%%' % (100 * prod / np.sum(v)))


if __name__ == '__main__':

    ### RUN TESTS
    testSoccer(1000)

    ### RUN PERFORMANCE TESTS
    #testSoccerPerformance()



    ### TO PROFILE ALGORITHM TIMING PERFORMANCE
    # import cProfile
    # cProfile.run('testSoccer(1000)')
    # cProfile.run('testOshiZumo(1000)')
