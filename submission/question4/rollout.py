"""
On-policy rollout buffer for PPO

**DO NOT CHANGE THIS FILE**
"""
from collections import namedtuple
from typing import Dict, Iterator

import numpy as np
import torch


Transition = namedtuple(
    "Transition", ("states", "actions", "rewards", "dones", "log_probs", "values")
)


class RolloutBuffer:
    """Fixed-horizon storage for on-policy rollouts collected across parallel environments

    Unlike the replay buffer used for DQN, this buffer is not sampled from during
    collection: it is filled once per rollout (``num_steps`` steps across ``num_envs``
    environments), handed to the agent's ``update`` for a number of epochs of minibatch
    updates, and then reset for the next rollout. It is pure storage/bookkeeping -
    the GAE and PPO loss maths live in the ``PPO`` agent, not here.

    :attr num_steps (int): rollout horizon (steps collected per environment before an update)
    :attr num_envs (int): number of parallel environments
    """

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int):
        """Constructor for a RolloutBuffer, initialising empty storage for one rollout

        :param num_steps (int): number of steps to collect per environment before an update
        :param num_envs (int): number of parallel environments being stepped simultaneously
        :param obs_dim (int): dimensionality of the flattened observation vector
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.states = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
    ):
        """Adds one step of transitions (across all parallel environments) to the buffer

        :param states (np.ndarray): observations of shape (num_envs, obs_dim) *before* this step
        :param actions (np.ndarray): actions of shape (num_envs,) taken in response to states
        :param rewards (np.ndarray): rewards of shape (num_envs,) received after taking actions
        :param dones (np.ndarray): terminal flags of shape (num_envs,)
        :param log_probs (np.ndarray): log pi(action|state) of shape (num_envs,) under the
            policy that was used to collect this step (the "old" policy in the PPO ratio)
        :param values (np.ndarray): V(state) of shape (num_envs,) estimated by the critic
            at collection time
        """
        t = self.ptr
        self.states[t] = states
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.ptr += 1

    def reset(self):
        """Resets the write pointer so the buffer can be filled with a new rollout

        Note: does not zero the underlying arrays, only marks them for overwriting -
        this is safe because a full rollout always overwrites every entry before
        ``get``/``minibatches`` are called again.
        """
        self.ptr = 0

    def get(self) -> Transition:
        """Returns the raw (num_steps, num_envs, ...) rollout data collected so far

        :return (Transition): namedtuple of the collected states/actions/rewards/dones/
            log_probs/values, each of shape (num_steps, num_envs, ...)
        """
        return Transition(
            self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values
        )

    def minibatches(
        self, advantages: np.ndarray, returns: np.ndarray, minibatch_size: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Flattens the rollout to (num_steps * num_envs, ...), shuffles it, and yields minibatches

        Intended to be called once per epoch inside ``PPO.update`` - call it again each
        epoch to get a fresh shuffle of the same rollout.

        :param advantages (np.ndarray): advantage estimates of shape (num_steps, num_envs),
            typically from ``PPO.compute_gae``
        :param returns (np.ndarray): value targets of shape (num_steps, num_envs)
        :param minibatch_size (int): number of transitions per yielded minibatch
        :return (Iterator[Dict[str, torch.Tensor]]): minibatches with keys "states",
            "actions", "old_log_probs", "advantages", "returns"
        """
        batch_size = self.num_steps * self.num_envs
        flat_states = self.states.reshape(batch_size, -1)
        flat_actions = self.actions.reshape(batch_size)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)

        indices = np.random.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            yield {
                "states": torch.from_numpy(flat_states[mb_idx]),
                "actions": torch.from_numpy(flat_actions[mb_idx]),
                "old_log_probs": torch.from_numpy(flat_log_probs[mb_idx]),
                "advantages": torch.from_numpy(flat_advantages[mb_idx]),
                "returns": torch.from_numpy(flat_returns[mb_idx]),
            }
