"""
2E) TD Learning: SARSA

This version of the algorithm follows Figure 6.9, Ch. 6.4 from Sutton, Barto

author: Luca G. Mc s1442231 -- UoE
"""

import numpy as np
from utils import *

# inits
gamma = 0.5 # like solutions sheet
alpha = 0.7
actions = 3
states = 4
eps = 0.1
num_episodes = 10
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

# Start of SARSA
Q = np.zeros((states, actions))
pi = get_pi_with_respect_Q(Q, states, actions, eps)
print('SARSA')
for i in range(1, num_episodes+1):
    print('\nEpisode {}'.format(i))
    state = random.randint(0, 2) # random start state
    action = np.random.choice(actions, p=pi[state]) # e-soft policy
    while state != 3:
        reward = R[action, state]
        print('S={}, a={}, r={}'.format(state, action, reward))
        next_state = get_next_state(state, action, P)
        next_action = np.random.choice(actions, p=pi[next_state])# e-soft policy
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        pi = get_pi_with_respect_Q(Q, states, actions, eps)
        state = next_state
        action = next_action
        print('Q={}'.format(Q))
    print('pi={}'.format(pi))
    state = random.randint(0, 2) # random start
