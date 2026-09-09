import gymnasium as gym
import os.path
import pickle

from submission.question3.agents import Reinforce
from submission.question3.train_reinforce import play_episode, CARTPOLE_CONFIG, SWEEP_RESULTS_FILE_CARTPOLE
from submission.util.result_processing import get_best_saved_run

ENV = "CARTPOLE"
RENDER = True
EVAL_BEST_SWEEP_RUN = False  # True to evaluate the best run from a saved sweep .pkl instead


def evaluate(env: gym.Env, config, output: bool = True) -> float:
    """
    Restore a trained REINFORCE agent from ``config['save_filename']`` and evaluate its
    policy on ``env``, averaged over ``config['eval_episodes']`` episodes.

    :param env (gym.Env): environment to evaluate on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (float): mean evaluation return
    """
    agent = Reinforce(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    agent.restore(config['save_filename'])

    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    if ENV == "CARTPOLE":
        CONFIG = CARTPOLE_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_CARTPOLE
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    if EVAL_BEST_SWEEP_RUN and os.path.exists(SWEEP_RESULTS_FILE):
        results = pickle.load(open(SWEEP_RESULTS_FILE, 'rb'))
        best_run, best_run_filename = get_best_saved_run(results)
        print(f"Best run was {best_run_filename}")
        CONFIG.update(best_run.config)
        CONFIG['save_filename'] = best_run_filename
    else:
        print(f"Evaluating saved model {CONFIG['save_filename']}")
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
