"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import numpy as np

from submission.constants import Q1_CONSTANTS
from submission.question1 import MDP, Transition, ValueIteration, PolicyIteration


def _rock_jump_mdp() -> MDP:
    mdp = MDP()
    mdp.add_transition(
        #         start   action  end     prob  reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )
    return mdp


def test_value_iteration_shapes():
    mdp = _rock_jump_mdp()
    solver = ValueIteration(mdp, Q1_CONSTANTS["gamma"])
    policy, V = solver.solve()

    assert V.shape == (solver.state_dim,)
    assert policy.shape == (solver.state_dim, solver.action_dim)
    assert np.allclose(policy.sum(axis=1), 1.0)
    assert np.all(policy >= 0.0) and np.all(policy <= 1.0)


def test_value_iteration_decode_policy():
    mdp = _rock_jump_mdp()
    solver = ValueIteration(mdp, Q1_CONSTANTS["gamma"])
    policy, _ = solver.solve()
    decoded = solver.decode_policy(policy)

    assert set(decoded.keys()) == set(mdp.states)
    assert all(action in mdp.actions for action in decoded.values())


def test_policy_iteration_shapes():
    mdp = _rock_jump_mdp()
    solver = PolicyIteration(mdp, Q1_CONSTANTS["gamma"])
    policy, V = solver.solve()

    assert V.shape == (solver.state_dim,)
    assert policy.shape == (solver.state_dim, solver.action_dim)
    assert np.allclose(policy.sum(axis=1), 1.0)
    assert np.all(policy >= 0.0) and np.all(policy <= 1.0)


def test_policy_iteration_decode_policy():
    mdp = _rock_jump_mdp()
    solver = PolicyIteration(mdp, Q1_CONSTANTS["gamma"])
    policy, _ = solver.solve()
    decoded = solver.decode_policy(policy)

    assert set(decoded.keys()) == set(mdp.states)
    assert all(action in mdp.actions for action in decoded.values())
