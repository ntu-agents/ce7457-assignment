import gymnasium as gym
import os.path
import pickle

from submission.question4.agents import PPO
from submission.question4.train_ppo import evaluate, LUNARLANDER_CONFIG, SWEEP_RESULTS_FILE_LUNARLANDER
from submission.util.result_processing import get_best_saved_run

ENV = "LUNARLANDER"
RENDER = True
EVAL_BEST_SWEEP_RUN = False


def run_evaluation(env: gym.Env, config) -> float:
    """
    Restores a trained PPO agent from `config["save_filename"]` and evaluates its greedy
    policy on `env`

    :param env (gym.Env): environment to evaluate on
    :param config: configuration dictionary mapping configuration keys to values
    :return (float): mean evaluation return
    """
    agent = PPO(action_space=env.action_space, observation_space=env.observation_space, **config)
    save_filename = config["save_filename"]
    if not os.path.dirname(save_filename):
        save_filename = os.path.join(os.path.dirname(__file__), save_filename)
    agent.restore(save_filename)
    return evaluate(agent, env, episodes=config["eval_episodes"], max_steps=config["episode_length"])


if __name__ == "__main__":
    if ENV == "LUNARLANDER":
        CONFIG = LUNARLANDER_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_LUNARLANDER
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"], render_mode="human" if RENDER else None)
    if EVAL_BEST_SWEEP_RUN and os.path.exists(SWEEP_RESULTS_FILE):
        results = pickle.load(open(SWEEP_RESULTS_FILE, "rb"))
        best_run, best_run_filename = get_best_saved_run(results)
        print(f"Best run was {best_run_filename}")
        CONFIG.update(best_run.config)
        CONFIG["save_filename"] = best_run_filename
    else:
        print(f"Evaluating saved model {CONFIG['save_filename']}")
    returns = run_evaluation(env, CONFIG)
    print(returns)
    env.close()
