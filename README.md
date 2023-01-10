# Introduction

The goal of this assignment is to implement the Deep Q-Learning algorithm with a simple continuous time 1-pendulumn enviroment, using discrete control input.

## Folder organization

* display.py: connect to geppetto-viewer or webbrowser
* pendulumn.py: Create a continuous state simulation environment for a N-pendulum
* pendulumn_dci.py: Describe continuous state pendulum environment with discrete control input (derived from pendulumn.py)
* replay_buffer.py: create a replay buffer class, to store the experiences
* dqn_agent.py: create a deep-q-network learning agent using the tensorflow library
* dqn_main.py: main file for the dqn algorithm applied to a single pendulumn enviroment