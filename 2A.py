"""
2A) Policy Iteration

This version of the algorithm follows Figure 4.3, Ch. 4.3 from Sutton, Barto

author: Luca G. Mc s1442231 -- UoE
"""

import numpy as np
from utils import *

def policy_evaluation(states, V, P, R, pi):
    """ Evaluates a given policy pi using DP """
    print('\nRunning Policy Evaluation')
    delta = 0.2
    iter_ = 1
    while delta > 0.1 and iter_ < 50:
        delta = 0
        print('iter: {}'.format(iter_))
        for state in range(states-1):
            v = V[state]
            V[state] = bellman(state, V, P, R, pi, states, None, gamma)
            delta = max(delta, np.abs(v - V[state]))
            print('S={}, V[state]={}, delta={}'.format(state, V[state], delta))
        iter_ += 1
    print('\nConverged on V = {}'.format(V))
    return V

def policy_improvement(states, pi, V, P, R, actions, gamma):
    """ Improves a given policy pi using DP """
    print('\nRunning Policy Improvement')
    policy_stable = True
    for state in range(states-1):
        b = pi[state]
        bellmans = []
        for action in range(actions):
            bellmans.append(bellman(state, V, P, R, pi, states, action, gamma))
        print('S={} {}'.format(state, bellmans))
        pi[state] = np.argmax(bellmans)
        if pi[state] != b:
            policy_stable = False
            print('Policy unstable, back to Evaluation')
    return pi, policy_stable

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

# Start of Policy Iteration
print('Running Policy Iteration')
print('------------------------')
V = np.zeros(states)
pi = np.array([0, 1, 2]) # same as solutions
policy_stable = False
while policy_stable == False:
    V = policy_evaluation(states, V, P, R, pi)
    pi, policy_stable = policy_improvement(states, pi, V, P, R, actions, gamma)
print('\nFound an optimal policy:')
print('pi = {}'.format(pi))
print('V = {}'.format(V))
