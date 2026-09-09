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
    from submission.question4 import PPO, RolloutBuffer
    from submission.question4.train_ppo import (
        LUNARLANDER_CONFIG,
        LUNARLANDER_HPARAMS_CLIP,
        LUNARLANDER_HPARAMS_ENTROPY,
        LUNARLANDER_HPARAMS_GAE,
    )

def test_config_0():
    from submission.question4.train_ppo import LUNARLANDER_CONFIG
    assert "eval_freq" in LUNARLANDER_CONFIG
    assert "eval_episodes" in LUNARLANDER_CONFIG
    assert "episode_length" in LUNARLANDER_CONFIG
    assert "max_timesteps" in LUNARLANDER_CONFIG

    assert "num_envs" in LUNARLANDER_CONFIG
    assert "n_steps" in LUNARLANDER_CONFIG
    assert "gae_lambda" in LUNARLANDER_CONFIG
    assert "clip_eps" in LUNARLANDER_CONFIG
    assert "entropy_coef" in LUNARLANDER_CONFIG
    assert "save_filename" in LUNARLANDER_CONFIG


def test_hparam_sweeps_0():
    from submission.question4.train_ppo import (
        LUNARLANDER_HPARAMS_CLIP,
        LUNARLANDER_HPARAMS_ENTROPY,
        LUNARLANDER_HPARAMS_GAE,
    )

    assert LUNARLANDER_HPARAMS_GAE["gae_lambda"] == [0.5, 0.8, 0.95]
    assert LUNARLANDER_HPARAMS_ENTROPY["entropy_coef"] == [0.0, 0.01, 0.1]
    assert LUNARLANDER_HPARAMS_CLIP["clip_eps"] == [0.1, 0.2, 0.5]
