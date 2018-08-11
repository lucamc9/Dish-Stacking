"""
2B) Value Iteration

This version of the algorithm follows Figure 4.5, Ch. 4.4 from Sutton, Barto 

author: Luca G. Mc s1442231 -- UoE
"""

import numpy as np
from utils import *

# inits
gamma = 1
actions = 3
states = 4
a_0 = np.array([[0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
a_1 = np.array([[1, 0, 0, 0],
                [0, 0.1, 0.9, 0],
                [0, 0.1, 0.9, 0],
                [0, 0, 0, 1]])
a_2 = np.array([[0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1]])
P = np.array([a_0, a_1, a_2])
R = np.array([[0, -20, -20, 0], [0, -2, -2, 0], [0, 5, 10, 0]])

# Start of Value Iteration
print('\nRunning Value Iteration')
V = np.zeros(states)
pi = np.zeros(actions)
delta = 0.2
iter_ = 1
while delta > 0.1 and iter_ < 50:
    delta = 0
    print('iter: {}'.format(iter_))
    for state in range(states-1):
        v = V[state]
        bellmans = []
        for action in range(actions):
            bellmans.append(bellman(state, V, P, R, pi, states, action, gamma))
        V[state] = np.max(bellmans)
        pi[state] = np.argmax(bellmans)
        delta = max(delta, np.abs(v - V[state]))
        print('S={}, V[state]={}, delta={}'.format(state, V[state], delta))
    iter_ += 1
print('\nFound deterministic policy, pi = {}'.format(pi))
