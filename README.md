# MinimaxQ-Learning
Applying minimaxQ learning algorithm to 2 agents games

Implementation of minimaxQ algorithm as proposed in:
https://www.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf

Implementation of three games: Soccer, OshiZumo (also known as Alesia), and a biased version of Rock Paper Scissors.

Only console interface for games...

## Run

- See Makefile

| Symbol | Agent                  |
| ------ | ---------------------- |
| T      | Tabular Q-learner      |
| S      | Tabular Minimax Q-learner|
| M      | Deep Minimax Q-learner |
| Q      | Deep Q-learner         |
| R      | Random policy          |
| C      | Challenger             |

### Examples

```bash
make tabularQR
```

```bash
make tabularMR
```

```bash
make QR
```

```bash
make MM
```

### Results
- Videos in `/tmp/deepsoccerXX/gym`.
