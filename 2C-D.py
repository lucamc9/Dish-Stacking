"""
2C) On-Policy MC Control, using first-visit/every-visit and e-greedy policy selection

This version of the algorithm follows Figure 5.6, Ch. 5.4 from Sutton, Barto

author: Luca G. Mc s1442231 -- UoE
"""

import numpy as np
from utils import *

def generate_episode(pi, max_actions, R, P, states, actions, returns, first_visit=True):
    # Generates an episode following policy pi, returns the states visited and returns observed
    states_seen = []
    state = random.randint(0, 2) # random start state
    return_ = 0
    for i in range(max_actions):
        action = np.random.choice(actions, p=pi[state])# e-soft policy
        reward = R[action, state]
        return_ += reward
        print('S={}, action={}, reward={}'.format(state, action, reward))
        states_seen.append((state, action))
        state = get_next_state(state, action, P)
        if state == 3:
            break
    accounted_states = []
    for state, action in states_seen:
        if first_visit == True:
            if (state, action) not in accounted_states:
                returns[(state, action)].append(return_)
                accounted_states.append((state, action))
        else:
            returns[(state, action)].append(return_)
    return states_seen, returns

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
num_episodes = 10
Q = np.zeros((states, actions))

# policies recommended in the solutions sheet
p_1 = np.array([[0.1, 0.8, 0.1],[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
p_2 = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
p_3 = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
pos_pi = [p_1, p_2, p_3]
pi = p_2 # starting policy

# default is first-visit, set first_visit to False for every-visit
first_visit = True

# Returns(s, a) is implemented as a dictionary
returns = {}
for state in range(states-1):
    for action in range(actions):
        returns[(state, action)] = []

# Start of On-Policy Monte Carlo Control
max_actions = 3
eps = 0.1
print('On-Policy Monte Carlo Control')
for i in range(num_episodes):
    print('\nGenerating episode {}'.format(i))
    states_seen, returns = generate_episode(pi, max_actions, R, P, states, actions, returns, first_visit=first_visit)
    for state, action in states_seen:
        Q[state, action] = np.mean(returns[state, action])
    print('Q={}'.format(Q))
    pi = get_pi_with_respect_Q(Q, states, actions, eps)
    print('pi = {}'.format(pi))
