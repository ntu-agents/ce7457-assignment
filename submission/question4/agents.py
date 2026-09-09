from typing import Dict, Iterable, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from submission.question3.agents import Agent
from submission.question3.networks import FCNetwork
from submission.question4.rollout import RolloutBuffer, Transition


class PPO(Agent):
    """The PPO agent for Q4

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

    :attr actor (FCNetwork): fully connected network computing per-action logits for the policy
    :attr critic (FCNetwork): fully connected network computing the state-value estimate V(s)
    :attr policy_optim (torch.optim): PyTorch optimiser for the actor network
    :attr value_optim (torch.optim): PyTorch optimiser for the critic network
    :attr gamma (float): discount rate gamma
    :attr gae_lambda (float): GAE trace-decay parameter lambda, trading off bias for variance
        in the advantage estimate (lambda=0 is the one-step TD advantage, lambda=1 is the
        full Monte-Carlo advantage)
    :attr clip_eps (float): PPO clipping parameter epsilon for the surrogate objective
    :attr entropy_coef (float): weight of the entropy bonus in the actor's loss
    :attr n_epochs (int): number of passes over each collected rollout during ``update``
    :attr minibatch_size (int): size of the minibatches each epoch is split into
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        entropy_coef: float,
        n_epochs: int,
        minibatch_size: int,
        **kwargs,
    ):
        """The constructor of the PPO agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate used for both the actor and critic Adam
            optimisers
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for the actor and
            critic fully connected networks
        :param gamma (float): discount rate gamma
        :param gae_lambda (float): GAE trace-decay parameter lambda
        :param clip_eps (float): PPO clipping parameter epsilon
        :param entropy_coef (float): weight of the entropy bonus
        :param n_epochs (int): number of epochs of minibatch updates per rollout
        :param minibatch_size (int): size of each minibatch
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.actor = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )
        self.critic = FCNetwork((STATE_SIZE, *hidden_size, 1), output_activation=None)

        self.policy_optim = Adam(self.actor.parameters(), lr=learning_rate, eps=1e-3)
        self.value_optim = Adam(self.critic.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size

        # ###############################################
        self.saveables.update(
            {
                "actor": self.actor,
                "critic": self.critic,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        Called before every rollout (Q4's and Q5's training scripts). Use it to schedule
        hyperparameters - e.g. annealing ``self.learning_rate`` (update the ``lr`` of both
        optimisers) or ``self.clip_eps`` towards zero over training. A no-op is fine for
        Question 4; optionally implement a schedule here for Question 5.

        :param timestep (int): current timestep at the beginning of the rollout
        :param max_timestep (int): maximum timesteps the training loop will run for
        """
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        Sample an action from the categorical distribution parameterised by ``self.actor``'s
        output logits. ``obs`` can be a single observation, shape ``(obs_dim,)``, or a batch
        of observations, shape ``(num_envs, obs_dim)``.

        - if ``explore`` is ``True``: return ``(action, log_prob, value)``, where
          ``log_prob`` is log pi(action|obs) and ``value`` is V(obs).
        - if ``explore`` is ``False``: act greedily and return just ``action``.

        :param obs (np.ndarray): a single observation, or a batch of observations
        :param explore (bool): whether to sample (True) or act greedily (False)
        :return: ``(action, log_prob, value)`` if ``explore`` else ``action``
        """
        with torch.no_grad():
            ### PUT YOUR CODE HERE ###
            raise NotImplementedError("Needed for Q4")

    @staticmethod
    def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes advantages and value targets using Generalized Advantage Estimation

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        Implement the GAE equations from the assignment PDF. ``last_value`` is V(s_T) for the
        state immediately after the last collected step, used to bootstrap the final advantage
        since the rollout may be cut off before the episode terminates.

        :param rewards (np.ndarray): rewards of shape (num_steps, num_envs)
        :param values (np.ndarray): V(s_t) at collection time, shape (num_steps, num_envs)
        :param dones (np.ndarray): terminal flags of shape (num_steps, num_envs)
        :param last_value (np.ndarray): V(s_T) after the last collected step, shape (num_envs,)
        :param gamma (float): discount rate gamma
        :param gae_lambda (float): GAE trace-decay parameter lambda
        :return (Tuple[np.ndarray, np.ndarray]): (advantages, returns), each of shape
            (num_steps, num_envs)
        """
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q4")

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_eps: float,
    ) -> torch.Tensor:
        """Computes the PPO clipped surrogate objective (as a loss to minimise)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        Implement the clipped objective from the assignment PDF. Remember the sign: the paper's
        objective is maximised, but this should return a loss to be minimised.

        :param log_probs (torch.Tensor): log pi_theta(a|s) under the *current* policy, shape
            (batch_size,)
        :param old_log_probs (torch.Tensor): log pi_theta_old(a|s) at collection time, shape
            (batch_size,)
        :param advantages (torch.Tensor): advantage estimates A of shape (batch_size,)
        :param clip_eps (float): clipping parameter epsilon
        :return (torch.Tensor): scalar clipped surrogate policy loss
        """
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q4")

    def entropy_loss(self, dist: Categorical) -> torch.Tensor:
        """Computes the negative (mean) entropy of the policy's action distribution

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        Returns the *negative* mean entropy, so it's a loss to minimise, encouraging exploration.

        :param dist (Categorical): the policy's action distribution for a batch of states
        :return (torch.Tensor): negative mean entropy of ``dist`` over the batch
        """
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q4")

    def value_loss(self, values_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Computes the critic's mean-squared-error value loss

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        :param values_pred (torch.Tensor): V(s;phi) predicted by the critic for a batch of
            states, shape (batch_size,)
        :param returns (torch.Tensor): value targets R (from ``compute_gae``) for the same
            batch, shape (batch_size,)
        :return (torch.Tensor): scalar mean-squared-error loss between ``values_pred`` and
            ``returns``
        """
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q4")

    def update(self, buffer: RolloutBuffer, last_value: np.ndarray) -> Dict[str, float]:
        """Update function for PPO

        **YOU MUST IMPLEMENT THE MINIBATCH LOOP BELOW FOR Q4**

        This function is called once per collected rollout (every ``num_steps * num_envs``
        environment steps). Advantages/returns are computed once with ``compute_gae`` and
        normalised. For each minibatch, over ``self.n_epochs`` passes:
          - get ``p_loss``, ``ent_loss``, ``v_loss`` from ``ppo_loss``, ``entropy_loss``,
            ``value_loss``
        Logging into ``policy_losses``/``value_losses``/``entropies`` is provided below.

        :param buffer (RolloutBuffer): rollout buffer holding one full rollout of experience,
        :param last_value (np.ndarray): V(s_T) of shape (num_envs,), the critic's estimate of
            the state immediately after the last step in the rollout.
        :return (Dict[str, float]): dictionary mapping from loss names to loss values, averaged
            over all minibatches/epochs of this update.
        """
        transition: Transition = buffer.get()
        advantages, returns = self.compute_gae(
            transition.rewards,
            transition.values,
            transition.dones,
            last_value,
            self.gamma,
            self.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, entropies = [], [], []
        for _ in range(self.n_epochs):
            for mb in buffer.minibatches(advantages, returns, self.minibatch_size):
                ### PUT YOUR CODE HERE ###
                # 1) Calculate loss for actor and critic
                #    p_loss = self.ppo_loss(...), ent_loss = self.entropy_loss(...)
                #    v_loss = self.value_loss(...)
                # 2) gradient update step on actor and critic
                
                raise NotImplementedError("Needed for Q4")

                # policy_losses.append(p_loss.item())
                # value_losses.append(v_loss.item())
                # entropies.append(-ent_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }
