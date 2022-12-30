import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
import time

from pendulumn_dci import Pendulum_dci
from dqn_agent import DQNagent
from replay_buffer import ReplayBuffer

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 5000          # Number of training episodes
NPRINT                  = 500           # print something every NPRINT episodes
MAX_EPISODE_LENGTH      = 100           # Max episode length
QVALUE_LEARNING_RATE = 1e-3
DISCOUNT                = 0.9           # Discount factor 
PLOT                    = True          # Plot stuff if True
exploration_prob                = 1     # initial exploration probability of eps-greedy policy
exploration_decreasing_decay    = 0.001 # exploration decay for exponential decreasing
min_exploration_prob            = 0.001 # minimum of exploration proba
nx = 2                                  # number of states
nu = 1                                  # number of control input

### --- Initializa agent, buffer and enviroment
agent = DQNagent(nx, nu, DISCOUNT, QVALUE_LEARNING_RATE)
buffer = ReplayBuffer()
env = Pendulum_dci(1)  # single pendulum

print("DONE !!")