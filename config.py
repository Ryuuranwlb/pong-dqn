from functools import partial

from DQN.dqn import my8DQNAgent, teacherDQNAgent


AGENT = {
    'DQN': my8DQNAgent,
    'DoubleDQN': partial(my8DQNAgent, double=True),
    'DuelingDQN': partial(my8DQNAgent, dueling=True),
    'D3QN': partial(my8DQNAgent, double=True, dueling=True),
    'TeacherDQN': teacherDQNAgent,
}
