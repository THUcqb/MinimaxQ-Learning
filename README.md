# MinimaxQ-Learning
Applying minimaxQ learning algorithm to 2 agents games

Implementation of minimaxQ algorithm as proposed in:
https://www.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf

Implementation of three games: Soccer, OshiZumo (also known as Alesia), and a biased version of Rock Paper Scissors.

Only console interface for games...

## Run

| Symbol | Agent                  |
| ------ | ---------------------- |
| M      | Deep Minimax Q-learner |
| Q      | Deep Q-learner         |
| R      | Random policy          |
| C      | Challenger             |

### Examples
- This will train 2 DQN playing against each other.
- All options: QR, QQ, MR, MM, MQ, QQC, MQC, QMC, MMC.

```bash
python run_dqn_deepsoccer.py --agents QQ
```

- This will train 2 DQN playing against each other. And the first one will be fixed after 5000000 iterations. We should expect that the first one will then be beaten (reward go down significantly).

```bash
python run_dqn_deepsoccer.py --agents QQC
```

- Now because we use max min instead of linear programming for Minimax Q. This may give the same result as QQC. The reason may be that the policy is still deterministic.
- However, in the paper, M won't be so bad when faced with a Q-challenger.

```bash
python run_dqn_deepsoccer.py --agents MQC
```

### Results
- Videos in `/tmp/deepsoccerXX/gym`.
