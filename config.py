from functools import partial

from DQN.dqn import DQNAgent

AGENT = {
    'DQN': DQNAgent,
    'DoubleDQN': partial(DQNAgent, double=True),
    'DuelingDQN': partial(DQNAgent, dueling=True),
    'D3QN': partial(DQNAgent, double=True, dueling=True),
}
