from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn
import torch.distributions as dist
from tools import *
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic

class MaskedPPOPolicy(PPOPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            optim: torch.optim.Optimizer,
            dist_fn: Type[torch.distributions.Distribution],
            eps_clip: float = 0.2,
            dual_clip: Optional[float] = None,
            value_clip: bool = False,
            advantage_normalization: bool = True,
            recompute_advantage: bool = False,
            k_placement: int = 80,
            num_bins: int = 3,
            **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self.k_placement = k_placement
        self.num_bins = num_bins
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:

        logits, hidden = self.actor(batch.obs["obs"], state=state)  
        mask = torch.FloatTensor(batch.obs["mask"]).to(logits.device)  
        
        # reshape logits from (batch_size, 2 * k_buffer * num_bins * k_placement) to (batch_size, num_bins * 2 * k_buffer * k_placement)
        x = logits.view(logits.shape[0], self.k_placement,-1) 
        x = x.view(logits.shape[0], -1, self.num_bins, self.k_placement)
        x = x.transpose(1, 2)  
        logits = x.contiguous().view(logits.shape[0],-1)

        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits=logits, masks=mask)
        torch.set_printoptions(precision=2, sci_mode=False)

        # print('logits:', logits)
        # print('mask:', mask)

        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = dist.probs.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
            else:
                raise Exception(f"Unsupported action_type: {self.action_type}")
        else:
            act = dist.sample()

        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn( # type: ignore
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps) # per-batch norm
                
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * minibatch.adv
                
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                # calculate loss for critic
                value = self.critic(minibatch.obs["obs"]).flatten() 
                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm: # clip large gradient
                    nn.utils.clip_grad_norm_(self._actor_critic.parameters(), max_norm=self._grad_norm)
                self.optim.step()

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

    def _compute_returns(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs["obs"]))
                v_s_.append(self.critic(minibatch.obs_next["obs"]))
        batch.v_s = torch.cat(v_s, dim=0).flatten()
         
        
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        

        if self._rew_norm:
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)

        unnormalized_returns, advantages = self.compute_episodic_return(
            batch, buffer, indices, v_s_, v_s, gamma=self._gamma, gae_lambda=self._lambda
        )
        # print('unnorm:', unnormalized_returns)
        # print(self._rew_norm)
        if self._rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns

        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch
