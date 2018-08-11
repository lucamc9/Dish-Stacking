"""
Functions needed to run the RL algorithms in 2A)-F)

author: Luca G. Mc s1442231 -- UoE
"""

import numpy as np
import random

def bellman(state, V, P, R, pi, states, action, gamma):
    """ Computes the Bellman equation for the state-value function V(s)"""
    if action == None:
        action = pi[state]
    v_sum = 0
    for pos_state in range(states):
        v_sum += P[action, state, pos_state] * (R[action, state] + gamma*V[pos_state])
    return v_sum

def get_next_state(state, action, P):
    """ Retrieves the next state given the current state, action taken
        and transition function P. In this case, I know a priori only
        states s_1 and s_2 have nonzero probability to transition to
        s_1 and s_2 """"
    if len(np.nonzero(P[action, state])[0]) > 1:
        idxs = np.nonzero(P[action, state])[0]
        rnd = random.uniform(0, 1)
        if rnd < P[action, state, idxs[0]]:
            next_state = idxs[0]
        else:
            next_state = idxs[1]
    else:
        next_state = np.argmax(P[action, state])
    return next_state

def get_pi_with_respect_Q(Q, states, actions, eps):
    """ Generate an e-greedy policy with respect to the action-value function Q """
    pi = np.zeros_like(Q)
    for state in range(states):
        action_max = np.argmax(Q[state, :])
        for action in range(actions):
            pi[state, action] = (1 - eps + eps/actions) if action == action_max else (eps/actions)
    return pi
