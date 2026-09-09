import copy
import os.path
import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np
import time
import torch
from tqdm import tqdm
from typing import Dict, Tuple

from submission.constants import Q4_LUNARLANDER_CONSTANTS as LUNARLANDER_CONSTANTS
from submission.question4.agents import PPO
from submission.question4.rollout import RolloutBuffer
from submission.util.hparam_sweeping import generate_hparam_configs
from submission.util.result_processing import Run

RENDER = False  # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False  # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 5  # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True  # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGHTS = False  # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "LUNARLANDER"

# DO NOT change the variable name LUNARLANDER_CONFIG, as we will depend on this
# to restore your trained model.
LUNARLANDER_CONFIG = {
    "eval_freq": 30000,
    "eval_episodes": 20,
    "learning_rate": 3e-4,
    "hidden_size": (64, 64),
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.01,
    "n_epochs": 4,
    "minibatch_size": 64,
    "num_envs": 8,
    "n_steps": 128,
}

LUNARLANDER_CONFIG.update(LUNARLANDER_CONSTANTS)

### ASSIGNMENT: SWEEP THESE TWO DICTS TO ANSWER QUESTIONS 4.1-4.2 IN answer_sheet.py ###
LUNARLANDER_HPARAMS_GAE = {
    "gae_lambda": [0.5, 0.8, 0.95],
}
LUNARLANDER_HPARAMS_ENTROPY = {
    "entropy_coef": [0.0, 0.01, 0.1],
}
LUNARLANDER_HPARAMS_CLIP = {
    "clip_eps": [0.1, 0.2, 0.5],
}

SWEEP_RESULTS_FILE_LUNARLANDER = "PPO-LunarLander-sweep-results.pkl"


def evaluate(agent: PPO, env: gym.Env, episodes: int, max_steps: int) -> float:
    """
    Evaluates the greedy policy of `agent` on `env`, averaged over a number of episodes

    :param agent (PPO): trained PPO agent to evaluate
    :param env (gym.Env): (non-vectorised) environment to evaluate on - construct it with
        render_mode="human" beforehand if you want to visualise the rollouts
    :param episodes (int): number of evaluation episodes
    :param max_steps (int): maximum number of steps per evaluation episode
    :return (float): mean return across evaluation episodes
    """
    returns = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done and steps < max_steps:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_return += reward
            steps += 1
        returns += ep_return / episodes
    return returns


def collect_rollout(
    vec_env: gym.vector.VectorEnv,
    agent: PPO,
    buffer: RolloutBuffer,
    obs: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, int]:
    """
    Collects one rollout of `n_steps` steps across all parallel environments in `vec_env`,
    storing every transition in `buffer` along the way

    :param vec_env (gym.vector.VectorEnv): vectorised environment to collect experience from
    :param agent (PPO): PPO agent used to act and to estimate values during collection
    :param buffer (RolloutBuffer): rollout buffer to fill (assumed freshly reset)
    :param obs (np.ndarray): observations to start collection from, of shape (num_envs, obs_dim)
    :param n_steps (int): number of steps to collect per environment
    :return (Tuple[np.ndarray, int]): observations after the last collected step (used to
        bootstrap the final advantage in `PPO.update`), and the number of environment steps
        collected (num_envs * n_steps)
    """
    num_envs = vec_env.num_envs
    for _ in range(n_steps):
        actions, log_probs, values = agent.act(obs, explore=True)
        next_obs, rewards, terminated, truncated, _ = vec_env.step(actions)
        dones = np.logical_or(terminated, truncated).astype(np.float32)
        buffer.add(obs, actions, rewards, dones, log_probs, values)
        obs = next_obs
    return obs, n_steps * num_envs


def train(
    vec_env: gym.vector.VectorEnv, eval_env: gym.Env, config, output: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of PPO on the given (vectorised) environment using the provided
    configuration

    :param vec_env (gym.vector.VectorEnv): vectorised environment to train on
    :param eval_env (gym.Env): single (non-vectorised) environment to evaluate on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]): average eval returns during
        training, evaluation timesteps, compute times at evaluation, and a dictionary
        containing other training metrics specific to PPO
    """
    agent = PPO(
        action_space=vec_env.single_action_space,
        observation_space=vec_env.single_observation_space,
        **config,
    )
    obs_dim = vec_env.single_observation_space.shape[0]
    buffer = RolloutBuffer(config["n_steps"], config["num_envs"], obs_dim)

    obs, _ = vec_env.reset()
    timesteps_elapsed = 0
    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            buffer.reset()
            obs, steps_collected = collect_rollout(vec_env, agent, buffer, obs, config["n_steps"])
            timesteps_elapsed += steps_collected
            pbar.update(steps_collected)

            last_value = (
                agent.critic(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
                .squeeze(-1)
                .detach()
                .numpy()
            )
            new_data = agent.update(buffer, last_value)
            for k, v in new_data.items():
                run_data[k].append(v)

            if timesteps_elapsed % config["eval_freq"] < steps_collected:
                eval_return = evaluate(
                    agent,
                    eval_env,
                    episodes=config["eval_episodes"],
                    max_steps=config["episode_length"],
                )
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                    )
                eval_returns_all.append(eval_return)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

    if config["save_filename"]:
        save_filename = config["save_filename"]
        if not os.path.dirname(save_filename):
            save_filename = os.path.join(os.path.dirname(__file__), save_filename)
        print("\nSaving to: ", agent.save(save_filename))

    run_data["train_updates"] = np.arange(1, len(run_data["policy_loss"]) + 1).tolist()

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":

    if ENV == "LUNARLANDER":
        CONFIG = LUNARLANDER_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_LUNARLANDER
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    vec_env = gym.make_vec(CONFIG["env"], num_envs=CONFIG["num_envs"], vectorization_mode="sync")
    eval_env = gym.make(CONFIG["env"], render_mode="human" if RENDER else None)

    if SWEEP:
        results = []
        for HPARAMS_SWEEP in (
            LUNARLANDER_HPARAMS_GAE,
            LUNARLANDER_HPARAMS_ENTROPY,
            LUNARLANDER_HPARAMS_CLIP,
        ):
            config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
            for config in config_list:
                if not SWEEP_SAVE_ALL_WEIGHTS:
                    config["save_filename"] = None
                run = Run(config)
                hparams_values = "_".join([":".join([key, str(config[key])]) for key in swept_params])
                run.run_name = hparams_values
                print("\nStarting new run...")
                for i in range(NUM_SEEDS_SWEEP):
                    print(f"\nTraining iteration: {i + 1}/{NUM_SEEDS_SWEEP}")
                    run_save_filename = "--".join([run.config["algo"], run.config["env"], hparams_values, str(i)])
                    if SWEEP_SAVE_ALL_WEIGHTS:
                        run.set_save_filename(run_save_filename)
                    eval_returns, eval_timesteps, times, run_data = train(vec_env, eval_env, run.config, output=False)
                    run.update(eval_returns, eval_timesteps, times, run_data)
                results.append(copy.deepcopy(run))
                print(
                    f"Finished run with hyperparameters {hparams_values}. "
                    f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}"
                )

        if SWEEP_SAVE_RESULTS:
            with open(SWEEP_RESULTS_FILE, "wb") as f:
                pickle.dump(results, f)
    else:
        _ = train(vec_env, eval_env, CONFIG)

    vec_env.close()
    eval_env.close()
