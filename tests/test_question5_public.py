"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""


def test_imports_0():
    from submission.question5.train_ppo import LUNARLANDER_CONFIG
    from submission.question5.evaluate_ppo import run_evaluation


def test_config_0():
    from submission.question5.train_ppo import LUNARLANDER_CONFIG

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


def test_save_filename_differs_from_question4():
    from submission.question4.train_ppo import LUNARLANDER_CONFIG as Q4_CONFIG
    from submission.question5.train_ppo import LUNARLANDER_CONFIG as Q5_CONFIG

    assert Q5_CONFIG["save_filename"] != Q4_CONFIG["save_filename"]
