"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gymnasium as gym
import os.path
import numpy as np

def test_imports_0():
    from submission.question3 import DQN, Reinforce, ReplayBuffer
    from submission.question3.train_dqn import MOUNTAINCAR_CONFIG as DQN_MOUNTAINCAR_CONFIG
    from submission.question3.train_reinforce import CARTPOLE_CONFIG as REINF_CARTPOLE_CONFIG

def test_config_0():
    from submission.question3.train_dqn import MOUNTAINCAR_CONFIG
    assert "eval_freq" in MOUNTAINCAR_CONFIG
    assert "eval_episodes" in MOUNTAINCAR_CONFIG
    assert "episode_length" in MOUNTAINCAR_CONFIG
    assert "max_timesteps" in MOUNTAINCAR_CONFIG

    assert "batch_size" in MOUNTAINCAR_CONFIG
    assert "buffer_capacity" in MOUNTAINCAR_CONFIG

def test_config_1():
    from submission.question3.train_reinforce import CARTPOLE_CONFIG
    assert "eval_freq" in CARTPOLE_CONFIG
    assert "eval_episodes" in CARTPOLE_CONFIG
    assert "episode_length" in CARTPOLE_CONFIG
    assert "max_timesteps" in CARTPOLE_CONFIG


