"""Model-Based Offline Policy Optimization (MOPO)."""

import collections
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Dict, Optional, Tuple
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.networks_and_utils_for_agent.mopo_nets_utils import ActorNetwork, CriticNetwork
from d2c.utils import policies, utils


class MOPOAgent(BaseAgent):
    """MOPO agent built on top of D2C's offline SAC-style agent pattern."""

    def __init__(
            self,
            rollout_freq: int = 1000,
            rollout_batch_size: int = 50000,
            rollout_length: int = 5,
            real_ratio: float = 0.05,
            reward_penalty_coef: float = 1.0,
            update_actor_freq: int = 1,
            alpha_multiplier: float = 1.0,
            alpha_init_value: float = 1.0,
            automatic_entropy_tuning: bool = True,
            target_entropy: float = -3.0,
            backup_entropy: bool = True,
            target_update_period: int = 1,
            grad_clip_norm: Optional[float] = 10.0,
            **kwargs: Any,
    ) -> None:
        self._rollout_freq = rollout_freq
        self._rollout_batch_size = rollout_batch_size
        self._rollout_length = rollout_length
        self._real_ratio = real_ratio
        self._reward_penalty_coef = reward_penalty_coef
        self._update_actor_freq = update_actor_freq
        self._alpha_multiplier = alpha_multiplier
        self._alpha_init_value = alpha_init_value
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._target_entropy = target_entropy
        self._backup_entropy = backup_entropy
        self._target_update_period = target_update_period
        self._grad_clip_norm = grad_clip_norm
        self._p_info = collections.OrderedDict()
        self._rollout_info = collections.OrderedDict()
        super(MOPOAgent, self).__init__(**kwargs)
        if self._target_entropy == 0.0:
            self._target_entropy = -float(self._a_dim)

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]

        def q_net_factory():
            return CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def p_net_factory():
            return ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        def log_alpha_net_factory():
            return torch.tensor(
                [math.log(self._alpha_init_value)],
                dtype=torch.float32,
                device=self._device,
            )

        return utils.Flags(
            p_net_factory=p_net_factory,
            q_net_factory=q_net_factory,
            n_q_fns=n_q_fns,
            log_alpha_net_factory=log_alpha_net_factory,
            device=self._device,
            automatic_entropy_tuning=self._automatic_entropy_tuning,
        )

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        if self._automatic_entropy_tuning:
            self._log_alpha_fn = self._agent_module.log_alpha_net

    def _init_vars(self) -> None:
        pass

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=self._q_fns.parameters(),
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_fn.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )
        if self._automatic_entropy_tuning:
            self._alpha_optimizer = utils.get_optimizer(opts.alpha[0])(
                parameters=[self._log_alpha_fn],
                lr=opts.alpha[1],
                weight_decay=self._weight_decays,
            )
            self._alpha = self._log_alpha_fn.exp() * self._alpha_multiplier
        else:
            self._alpha = torch.tensor(
                self._alpha_init_value * self._alpha_multiplier,
                dtype=torch.float32,
                device=self._device,
            )

    def _build_alpha_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        with torch.no_grad():
            _, _, log_pi = self._p_fn(states)
        alpha_loss = (-self._log_alpha_fn.exp() * (log_pi + self._target_entropy)).mean()
        self._alpha = self._log_alpha_fn.exp() * self._alpha_multiplier

        info = collections.OrderedDict()
        info['alpha'] = self._alpha.detach().mean()
        info['alpha_loss'] = alpha_loss.detach()
        return alpha_loss, info

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        discounts = batch['dsc']
        alpha = self._alpha.detach()

        with torch.no_grad():
            _, next_actions, next_log_pi = self._p_fn(next_states)
            target_q1 = self._q_target_fns[0](next_states, next_actions)
            target_q2 = self._q_target_fns[1](next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            if self._backup_entropy:
                target_q = target_q - alpha * next_log_pi
            td_target = rewards + discounts * self._discount * target_q

        q1_pred = self._q_fns[0](states, actions)
        q2_pred = self._q_fns[1](states, actions)
        q1_loss = F.mse_loss(q1_pred, td_target)
        q2_loss = F.mse_loss(q2_pred, td_target)
        q_loss = q1_loss + q2_loss

        info = collections.OrderedDict()
        info['Q1'] = q1_pred.detach().mean()
        info['Q2'] = q2_pred.detach().mean()
        info['Q_target'] = td_target.detach().mean()
        info['Q1_loss'] = q1_loss.detach()
        info['Q2_loss'] = q2_loss.detach()
        info['Q_loss'] = q_loss.detach()
        info['reward_mean'] = rewards.detach().mean()
        return q_loss, info

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        alpha = self._alpha.detach()

        _, sampled_actions, log_pi = self._p_fn(states)
        q1_pi = self._q_fns[0](states, sampled_actions)
        q2_pi = self._q_fns[1](states, sampled_actions)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        p_loss = (alpha * log_pi - min_q_pi).mean()

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss.detach()
        info['log_pi'] = log_pi.detach().mean()
        info['Q_in_actor_loss'] = min_q_pi.detach().mean()
        return p_loss, info

    def _optimize_q(self, batch: Dict) -> Dict:
        q_loss, info = self._build_q_loss(batch)
        self._q_optimizer.zero_grad()
        q_loss.backward()
        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._q_fns.parameters(), self._grad_clip_norm)
        self._q_optimizer.step()
        return info

    def _optimize_p(self, batch: Dict) -> Dict:
        p_loss, info = self._build_p_loss(batch)
        self._p_optimizer.zero_grad()
        p_loss.backward()
        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._p_fn.parameters(), self._grad_clip_norm)
        self._p_optimizer.step()
        return info

    def _optimize_alpha(self, batch: Dict) -> Dict:
        alpha_loss, info = self._build_alpha_loss(batch)
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([self._log_alpha_fn], self._grad_clip_norm)
        self._alpha_optimizer.step()
        return info

    @staticmethod
    def _cat_batch(real_batch: Optional[Dict], model_batch: Optional[Dict]) -> Dict:
        if real_batch is None:
            return model_batch
        if model_batch is None:
            return real_batch
        mixed_batch = collections.OrderedDict()
        for key in real_batch.keys():
            mixed_batch[key] = torch.cat([real_batch[key], model_batch[key]], dim=0)
        return mixed_batch

    def _rollout_transitions(self) -> None:
        if self._empty_dataset is None or self._env.dynamics_model is None:
            return
        init_batch = self._train_data.sample_batch(self._rollout_batch_size)
        observations = init_batch['s1']
        rollout_steps = 0
        total_transitions = 0
        penalty_means = []
        reward_means = []

        for _ in range(self._rollout_length):
            with torch.no_grad():
                _, actions, _ = self._p_fn(observations)
                next_obs, rewards, terminals, info = self._env.dynamics_model.predict(
                    observations,
                    actions,
                    reward_penalty_coef=self._reward_penalty_coef,
                )
                _, next_actions, _ = self._p_fn(next_obs)
            self._empty_dataset.add_transitions(
                state=observations,
                action=actions,
                next_state=next_obs,
                next_action=next_actions,
                reward=rewards,
                done=terminals,
            )
            rollout_steps += 1
            total_transitions += observations.shape[0]
            penalty_means.append(info['penalty'].detach().mean())
            reward_means.append(info['raw_rewards'].detach().mean())
            nonterm_mask = terminals < 0.5
            if torch.sum(nonterm_mask) == 0:
                break
            observations = next_obs[nonterm_mask]

        if rollout_steps > 0:
            self._rollout_info['model_rollout_steps'] = torch.tensor(
                float(rollout_steps),
                dtype=torch.float32,
                device=self._device,
            )
            self._rollout_info['model_rollout_transitions'] = torch.tensor(
                float(total_transitions),
                dtype=torch.float32,
                device=self._device,
            )
            self._rollout_info['model_buffer_size'] = torch.tensor(
                float(self._empty_dataset.size),
                dtype=torch.float32,
                device=self._device,
            )
            self._rollout_info['model_penalty'] = torch.stack(penalty_means).mean()
            self._rollout_info['model_reward'] = torch.stack(reward_means).mean()

    def _get_train_batch(self) -> Dict:
        if self._global_step % self._rollout_freq == 0:
            self._rollout_transitions()

        real_batch_size = int(self._batch_size * self._real_ratio)
        real_batch_size = min(real_batch_size, self._batch_size)
        model_batch_size = self._batch_size - real_batch_size

        if self._empty_dataset is None or self._empty_dataset.size == 0 or model_batch_size == 0:
            return self._train_data.sample_batch(self._batch_size)

        real_batch = None
        model_batch = None
        if real_batch_size > 0:
            real_batch = self._train_data.sample_batch(real_batch_size)
        if model_batch_size > 0:
            model_batch = self._empty_dataset.sample_batch(model_batch_size)
        return self._cat_batch(real_batch, model_batch)

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()
        q_info = self._optimize_q(batch)
        info.update(q_info)

        if self._global_step % self._update_actor_freq == 0:
            self._p_info = self._optimize_p(batch)
            info.update(self._p_info)
            if self._automatic_entropy_tuning:
                alpha_info = self._optimize_alpha(batch)
                info.update(alpha_info)

        if self._global_step % self._target_update_period == 0:
            self._update_target_fns(self._q_fns, self._q_target_fns)

        info.update(self._rollout_info)
        return info

    def _build_test_policies(self) -> None:
        self._test_policies['main'] = policies.DeterministicSoftPolicy(a_network=self._p_fn)

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(
            torch.load(ckpt_name + '.pth', map_location=self._device, weights_only=True)
        )


class AgentModule(BaseAgentModule):
    """Container of trainable MOPO modules."""

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)
        self._p_net = self._net_modules.p_net_factory().to(device)
        if self._net_modules.automatic_entropy_tuning:
            self._log_alpha_net = nn.Parameter(self._net_modules.log_alpha_net_factory())

    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets

    @property
    def p_net(self) -> nn.Module:
        return self._p_net

    @property
    def log_alpha_net(self) -> nn.Parameter:
        return self._log_alpha_net
