Q1_CONSTANTS = {
    "gamma": 0.85,
}

Q2_CONSTANTS = {
    "env": "FrozenLake8x8-v1",
    "eps_max_steps": 200,
    "eval_episodes": 500,
    "eval_eps_max_steps": 200,
}

Q2_MC_CONSTANTS = Q2_CONSTANTS.copy()
Q2_MC_CONSTANTS["total_eps"] = 300000

Q2_QL_CONSTANTS = Q2_CONSTANTS.copy()
Q2_QL_CONSTANTS["total_eps"] = 10000

Q3_CARTPOLE_CONSTANTS = {
    "env": "CartPole-v0",
    "gamma": 0.99,
    "episode_length": 200,
    "max_time": 30 * 60,
    "save_filename": None,
    "algo": None,
}

Q3_DQN_CARTPOLE_CONSTANTS = Q3_CARTPOLE_CONSTANTS.copy()
Q3_DQN_CARTPOLE_CONSTANTS["max_timesteps"] = 40000
Q3_DQN_CARTPOLE_CONSTANTS["algo"] = "DQN"

Q3_REINFORCE_CARTPOLE_CONSTANTS = Q3_CARTPOLE_CONSTANTS.copy()
Q3_REINFORCE_CARTPOLE_CONSTANTS["max_timesteps"] = 500000
Q3_REINFORCE_CARTPOLE_CONSTANTS["algo"] = "Reinforce"

Q3_MOUNTAINCAR_CONSTANTS = {
    "env": "MountainCar-v0",
    "gamma": 0.99,
    "episode_length": 200,
    "max_time": 120 * 60,
    "save_filename": None,
    "algo": None,
}

Q3_DQN_MOUNTAINCAR_CONSTANTS = Q3_MOUNTAINCAR_CONSTANTS.copy()
Q3_DQN_MOUNTAINCAR_CONSTANTS["max_timesteps"] = 700000
Q3_DQN_MOUNTAINCAR_CONSTANTS["algo"] = "DQN"

Q4_LUNARLANDER_CONSTANTS = {
    "env": "LunarLander-v3",
    "gamma": 0.99,
    "episode_length": 1000,
    "max_time": 60 * 60,
    "max_timesteps": 1_500_000,
    "save_filename": "lunarlander_latest.pt",
    "algo": "PPO",
}

# Question 5 reuses Q4's PPO agent
Q5_LUNARLANDER_CONSTANTS = Q4_LUNARLANDER_CONSTANTS.copy()
Q5_LUNARLANDER_CONSTANTS["save_filename"] = "lunarlander_hparam_latest.pt"
