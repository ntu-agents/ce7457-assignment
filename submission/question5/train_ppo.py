import os.path

import gymnasium as gym

from submission.constants import Q5_LUNARLANDER_CONSTANTS as LUNARLANDER_CONSTANTS
from submission.question4.train_ppo import train
from submission.util.hparam_sweeping import generate_hparam_configs, grid_search, random_search
from submission.util.result_processing import Run

RENDER = False  # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False  # TRUE TO SWEEP OVER THE HYPERPARAMETER CONFIGURATIONS DEFINED BELOW
NUM_SEEDS_SWEEP = 3  # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True  # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGHTS = False  # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "LUNARLANDER"

### ASSIGNMENT: EDIT THIS CONFIG (or PPO.schedule_hyperparameters) TO MAXIMISE PPO'S
### PERFORMANCE ON LUNARLANDER FOR QUESTION 5. Values below are Q4's defaults, as a starting point.
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
# mkae so training this script saves model into question5/, not question4/.
LUNARLANDER_CONFIG["save_filename"] = os.path.join(
    os.path.dirname(__file__), LUNARLANDER_CONSTANTS["save_filename"]
)

### ASSIGNMENT: OPTIONALLY DEFINE A HYPERPARAMETER SWEEP HERE TO ANSWER QUESTION 5.1 ###
LUNARLANDER_HPARAMS_SWEEP = {}

SWEEP_RESULTS_FILE_LUNARLANDER = "PPO-LunarLander-hparam-sweep-results.pkl"


if __name__ == "__main__":

    if ENV == "LUNARLANDER":
        CONFIG = LUNARLANDER_CONFIG
        HPARAMS_SWEEP = LUNARLANDER_HPARAMS_SWEEP
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_LUNARLANDER
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    vec_env = gym.make_vec(CONFIG["env"], num_envs=CONFIG["num_envs"], vectorization_mode="sync")
    eval_env = gym.make(CONFIG["env"], render_mode="human" if RENDER else None)

    if SWEEP:
        import copy
        import pickle

        results = []
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
                # Resolved against this file's directory (not question4/train_ppo.py's, where
                # the shared `train()` we call is defined) so sweep checkpoints land in
                # question5/ alongside lunarlander_hparam_latest.pt.
                run_save_filename = os.path.join(
                    os.path.dirname(__file__),
                    "--".join([run.config["algo"], run.config["env"], hparams_values, str(i)]),
                )
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
