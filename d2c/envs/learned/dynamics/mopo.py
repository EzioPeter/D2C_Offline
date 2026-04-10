"""MOPO-style ensemble dynamics with elite selection and holdout training."""

import collections
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Normal
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, ClassVar
from d2c.envs.learned.dynamics.base import BaseDyna, BaseDynaModule
from d2c.envs.learned.dynamics.mopo_terminals import get_mopo_terminal_fn
from d2c.utils import utils


class StandardNormalizer:
    """A light-weight normalizer matching the reference MOPO behavior."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mean = None
        self.var = None
        self.tot_count = 0

    def update(self, samples: Tensor) -> None:
        """Accumulate dataset statistics for logging/debugging."""
        sample_count = len(samples)
        if self.tot_count == 0:
            dim = samples.shape[1]
            self.mean = torch.zeros((1, dim), dtype=torch.float32, device=samples.device)
            self.var = torch.ones((1, dim), dtype=torch.float32, device=samples.device)

        sample_mean = torch.mean(samples, dim=0, keepdims=True)
        sample_var = torch.var(samples, dim=0, keepdims=True, unbiased=False)
        delta_mean = sample_mean - self.mean
        total_count = self.tot_count + sample_count
        self.mean = self.mean + delta_mean * sample_count / total_count
        prev_var = self.var * self.tot_count
        curr_var = sample_var * sample_count
        self.var = (
            prev_var
            + curr_var
            + delta_mean * delta_mean * self.tot_count * sample_count / total_count
        ) / total_count
        self.var[self.var < 1e-12] = 1.0
        self.tot_count = total_count

    def transform(self, data: Tensor) -> Tensor:
        """Keep identity transform to match the reference repository quirk."""
        return data


class Swish(nn.Module):
    """Swish activation."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.sigmoid(inputs)


def get_activation_cls(act_fn_name: str) -> Type[nn.Module]:
    """Return an activation class by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name == 'tanh':
        return nn.Tanh
    if act_fn_name == 'sigmoid':
        return nn.Sigmoid
    if act_fn_name == 'relu':
        return nn.ReLU
    if act_fn_name == 'identity':
        return nn.Identity
    if act_fn_name == 'swish':
        return Swish
    raise NotImplementedError(f'Activation function {act_fn_name} is not implemented.')


class EnsembleMLP(nn.Module):
    """Single member of the MOPO ensemble."""

    def __init__(
            self,
            input_dim: int,
            out_dim: int,
            hidden_dims: Sequence[int],
            act_fn: str = 'swish',
            out_act_fn: str = 'identity',
    ) -> None:
        super(EnsembleMLP, self).__init__()
        hidden_sizes = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        act_cls = get_activation_cls(act_fn)
        out_act_cls = get_activation_cls(out_act_fn)
        for in_dim, out_dim_ in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.extend([nn.Linear(in_dim, out_dim_), act_cls()])
        layers.extend([nn.Linear(hidden_sizes[-1], out_dim), out_act_cls()])
        self._network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self._network(inputs)

    @property
    def weights(self) -> List[Tensor]:
        """Linear weights used for weight decay."""
        return [m.weight for m in self._network if isinstance(m, nn.Linear)]


class EnsembleDynamicsModel(nn.Module):
    """Ensemble transition model used by MOPO."""

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dims: Sequence[int],
            ensemble_size: int = 7,
            num_elite: int = 5,
            decay_weights: Optional[Sequence[float]] = None,
            act_fn: str = 'swish',
            out_act_fn: str = 'identity',
            reward_dim: int = 1,
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(EnsembleDynamicsModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.out_dim = obs_dim + reward_dim
        self.ensemble_size = ensemble_size
        self.num_elite = num_elite
        self.decay_weights = decay_weights
        models = []
        for _ in range(ensemble_size):
            models.append(
                EnsembleMLP(
                    input_dim=obs_dim + action_dim,
                    out_dim=self.out_dim * 2,
                    hidden_dims=hidden_dims,
                    act_fn=act_fn,
                    out_act_fn=out_act_fn,
                )
            )
        self.ensemble_models = nn.ModuleList(models)
        self.register_buffer(
            'elite_model_idxes',
            torch.arange(num_elite, dtype=torch.long, device=device),
        )
        init_max = torch.ones((1, self.out_dim), dtype=torch.float32, device=device) / 2
        init_min = -torch.ones((1, self.out_dim), dtype=torch.float32, device=device) * 10
        self.max_logvar = nn.Parameter(init_max)
        self.min_logvar = nn.Parameter(init_min)
        self.to(device)

    def predict(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Return mean/logvar for all ensemble members."""
        if inputs.dim() == 3:
            outputs = [net(member_inputs) for member_inputs, net in zip(torch.unbind(inputs), self.ensemble_models)]
        else:
            outputs = [net(inputs) for net in self.ensemble_models]
        predictions = torch.stack(outputs, dim=0)
        mean = predictions[..., :self.out_dim]
        logvar = predictions[..., self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_decay_loss(self) -> Tensor:
        """Compute ensemble weight decay loss."""
        if self.decay_weights is None:
            return torch.tensor(0.0, device=self.max_logvar.device)
        losses = []
        for model_net in self.ensemble_models:
            losses.append(
                sum(
                    decay * torch.sum(weight.square())
                    for decay, weight in zip(self.decay_weights, model_net.weights)
                )
            )
        return torch.sum(torch.stack(losses))

    def set_elite_model_indices(self, elite_indices: Sequence[int]) -> None:
        elite_tensor = torch.as_tensor(
            elite_indices,
            dtype=torch.long,
            device=self.elite_model_idxes.device,
        )
        self.elite_model_idxes.copy_(elite_tensor)

    def load_state_dicts(self, state_dicts: Sequence[Dict[str, Tensor]]) -> None:
        for idx, state_dict in enumerate(state_dicts):
            self.ensemble_models[idx].load_state_dict(state_dict)


class MopoDyna(BaseDyna):
    """MOPO ensemble dynamics with holdout-based elite selection."""

    TYPE: ClassVar[str] = 'mopo'

    def __init__(
            self,
            holdout_ratio: float = 0.1,
            inc_var_loss: bool = True,
            use_weight_decay: bool = True,
            max_model_update_epochs_to_improve: int = 5,
            max_model_train_iterations: Optional[int] = None,
            max_train_steps: int = 800000,
            env_name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        self._holdout_ratio = holdout_ratio
        self._inc_var_loss = inc_var_loss
        self._use_weight_decay = use_weight_decay
        self._max_model_update_epochs_to_improve = max_model_update_epochs_to_improve
        self._max_model_train_iterations = (
            math.inf if max_model_train_iterations in (None, 'None') else max_model_train_iterations
        )
        self._max_train_steps = max_train_steps
        self._env_name = env_name
        self._terminal_fn = get_mopo_terminal_fn(env_name) if env_name is not None else None
        super(MopoDyna, self).__init__(**kwargs)

    def _get_modules(self) -> utils.Flags:
        model_params = self._model_params

        def d_net_factory() -> EnsembleDynamicsModel:
            return EnsembleDynamicsModel(
                obs_dim=self._state_dim,
                action_dim=self._action_dim,
                hidden_dims=model_params.hidden_dims,
                ensemble_size=model_params.ensemble_size,
                num_elite=model_params.num_elite,
                decay_weights=model_params.decay_weights,
                act_fn=model_params.act_fn,
                out_act_fn=model_params.out_act_fn,
                reward_dim=1 if self._with_reward else 0,
                device=self._device,
            )

        return utils.Flags(
            d_net_factory=d_net_factory,
            device=self._device,
        )

    def _build_fns(self) -> None:
        self._dyna_module = DynaModule(modules=self._modules)
        self._d_fns = self._dyna_module.d_net

    def _init_vars(self) -> None:
        self._obs_normalizer = StandardNormalizer()
        self._act_normalizer = StandardNormalizer()
        self._finished = False
        self._epoch_indices = None
        self._epoch_cursor = 0
        self._epochs_since_update = 0
        self._model_train_epochs = 0
        self._last_holdout_mse = None
        self._best_snapshot_losses = None
        self._model_best_snapshots = None

    def _build_optimizers(self) -> None:
        opt = self._optimizers
        self._optimizer = utils.get_optimizer(opt[0])(
            parameters=self._d_fns.parameters(),
            lr=opt[1],
            weight_decay=self._weight_decays,
        )

    def _train_test_split(self) -> None:
        shuffle_indices = self._train_data.shuffle_indices
        full_size = self._train_data.size
        train_size = int(full_size * (1 - self._holdout_ratio))
        train_size = min(max(train_size, 1), max(full_size - 1, 1))
        self._train_indices = shuffle_indices[:train_size]
        self._test_indices = shuffle_indices[train_size:]
        if len(self._test_indices) == 0:
            self._test_indices = shuffle_indices[-1:]
            self._train_indices = shuffle_indices[:-1]
        all_data = self._train_data.data
        self._obs_normalizer.reset()
        self._act_normalizer.reset()
        self._obs_normalizer.update(all_data['s1'][self._train_indices])
        self._act_normalizer.update(all_data['a1'][self._train_indices])
        self.reset_best_snapshots()
        eval_mse_losses, _ = self.eval_data(update_elite_models=False)
        self.update_best_snapshots(eval_mse_losses)
        self._last_holdout_mse = eval_mse_losses
        self._reset_train_epoch()

    def _reset_train_epoch(self) -> None:
        epoch_indices = np.array(self._train_indices, copy=True)
        np.random.shuffle(epoch_indices)
        self._epoch_indices = epoch_indices
        self._epoch_cursor = 0

    def _get_train_batch(self) -> Dict:
        if self._epoch_indices is None or self._epoch_cursor >= len(self._epoch_indices):
            if self._finished:
                return self._train_data.get_batch_indices(self._train_indices[:self._batch_size])
            self._reset_train_epoch()
        batch_end = min(self._epoch_cursor + self._batch_size, len(self._epoch_indices))
        batch_indices = self._epoch_indices[self._epoch_cursor:batch_end]
        self._epoch_cursor = batch_end
        return self._train_data.get_batch_indices(batch_indices)

    def _build_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        a1 = batch['a1']
        s2 = batch['s2']
        reward = batch['reward'].unsqueeze(-1) if self._with_reward else None
        targets = s2 - s1
        if reward is not None:
            targets = torch.cat([targets, reward], dim=-1)
        inputs = torch.cat(self.transform_obs_action(s1, a1), dim=-1)
        pred_means, pred_logvars = self._d_fns.predict(inputs)
        train_mse_losses, train_var_losses = self._model_loss(pred_means, pred_logvars, targets)
        train_mse_loss = torch.sum(train_mse_losses)
        train_var_loss = torch.sum(train_var_losses)
        total_loss = train_mse_loss + train_var_loss
        total_loss = total_loss + 0.01 * torch.sum(self._d_fns.max_logvar)
        total_loss = total_loss - 0.01 * torch.sum(self._d_fns.min_logvar)
        decay_loss = torch.tensor(0.0, device=self._device)
        if self._use_weight_decay:
            decay_loss = self._d_fns.get_decay_loss()
            total_loss = total_loss + decay_loss
        info = collections.OrderedDict()
        info['loss/model_train_mse'] = train_mse_loss.detach()
        info['loss/model_train_var'] = train_var_loss.detach()
        info['loss/model_train'] = total_loss.detach()
        info['loss/model_decay'] = decay_loss.detach()
        info['misc/max_logvar'] = self._d_fns.max_logvar.detach().mean()
        info['misc/min_logvar'] = self._d_fns.min_logvar.detach().mean()
        return total_loss, info

    def _build_test_loss(self, batch: Dict) -> Dict:
        del batch
        eval_mse_losses, _ = self.eval_data(update_elite_models=False)
        info = collections.OrderedDict()
        info['loss/model_holdout_mse'] = torch.as_tensor(
            eval_mse_losses.mean(),
            dtype=torch.float32,
            device=self._device,
        )
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        loss, info = self._build_loss(batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return info

    def _model_loss(
            self,
            pred_means: Tensor,
            pred_logvars: Tensor,
            groundtruths: Tensor,
            mse_only: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        target = groundtruths.unsqueeze(0)
        if mse_only:
            mse_losses = torch.mean((pred_means - target) ** 2, dim=(1, 2))
            return mse_losses, None
        if not self._inc_var_loss:
            raise NotImplementedError('MOPO dynamics expects inc_var_loss=True for faithful training.')
        inv_var = torch.exp(-pred_logvars)
        mse_losses = torch.mean(torch.mean((pred_means - target) ** 2 * inv_var, dim=-1), dim=-1)
        var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        return mse_losses, var_losses

    def _finish_epoch(self) -> None:
        eval_mse_losses, _ = self.eval_data(update_elite_models=False)
        updated = self.update_best_snapshots(eval_mse_losses)
        self._epochs_since_update += 1
        if updated:
            self._model_train_epochs += self._epochs_since_update
            self._epochs_since_update = 0
        self._last_holdout_mse = eval_mse_losses
        self._train_info['loss/model_holdout_mse'] = float(eval_mse_losses.mean())
        self._train_info['misc/model_train_epochs'] = float(self._model_train_epochs)
        self._train_info['misc/model_train_steps'] = float(self._global_step)
        self._train_info['misc/norm_obs_mean'] = float(self._obs_normalizer.mean.mean().detach().cpu())
        self._train_info['misc/norm_obs_var'] = float(self._obs_normalizer.var.mean().detach().cpu())
        self._train_info['misc/norm_act_mean'] = float(self._act_normalizer.mean.mean().detach().cpu())
        self._train_info['misc/norm_act_var'] = float(self._act_normalizer.var.mean().detach().cpu())
        if self._epochs_since_update >= self._max_model_update_epochs_to_improve \
                or self._global_step >= self._max_model_train_iterations \
                or self._global_step >= self._max_train_steps:
            self._finalize_training()

    def _finalize_training(self) -> None:
        self.load_best_snapshots()
        self.eval_data(update_elite_models=True)
        self._finished = True

    def train_step(self) -> None:
        if self._finished:
            return
        train_batch = self._get_train_batch()
        info = self._optimize_step(train_batch)
        for key, val in info.items():
            self._train_info[key] = val.item() if isinstance(val, Tensor) else val
        self._global_step += 1
        if self._epoch_cursor >= len(self._epoch_indices):
            self._finish_epoch()

    def test_step(self) -> None:
        info = self._build_test_loss({})
        for key, val in info.items():
            self._train_info[key] = val.item() if isinstance(val, Tensor) else val

    def reset_best_snapshots(self) -> None:
        self._model_best_snapshots = [
            copy.deepcopy(self._d_fns.ensemble_models[idx].state_dict())
            for idx in range(self._d_fns.ensemble_size)
        ]
        self._best_snapshot_losses = [1e10 for _ in range(self._d_fns.ensemble_size)]

    def update_best_snapshots(self, val_losses: np.ndarray) -> bool:
        updated = False
        for idx, current_loss in enumerate(val_losses):
            best_loss = self._best_snapshot_losses[idx]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self._best_snapshot_losses[idx] = current_loss
                self._model_best_snapshots[idx] = copy.deepcopy(
                    self._d_fns.ensemble_models[idx].state_dict()
                )
                updated = True
        return updated

    def load_best_snapshots(self) -> None:
        self._d_fns.load_state_dicts(self._model_best_snapshots)

    def reset_normalizers(self) -> None:
        self._obs_normalizer.reset()
        self._act_normalizer.reset()

    def update_normalizer(self, obs: Tensor, action: Tensor) -> None:
        self._obs_normalizer.update(obs)
        self._act_normalizer.update(action)

    def transform_obs_action(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        return self._obs_normalizer.transform(obs), self._act_normalizer.transform(action)

    @torch.no_grad()
    def eval_data(self, update_elite_models: bool = False) -> Tuple[np.ndarray, None]:
        test_batch = self._train_data.get_batch_indices(self._test_indices)
        s1 = test_batch['s1']
        a1 = test_batch['a1']
        s2 = test_batch['s2']
        reward = test_batch['reward'].unsqueeze(-1) if self._with_reward else None
        targets = s2 - s1
        if reward is not None:
            targets = torch.cat([targets, reward], dim=-1)
        inputs = torch.cat(self.transform_obs_action(s1, a1), dim=-1)

        pred_means_parts = []
        pred_logvars_parts = []
        data_size = inputs.shape[0]
        for start in range(0, data_size, self._batch_size):
            stop = min(start + self._batch_size, data_size)
            mean_part, logvar_part = self._d_fns.predict(inputs[start:stop])
            pred_means_parts.append(mean_part)
            pred_logvars_parts.append(logvar_part)
        pred_means = torch.cat(pred_means_parts, dim=1)
        pred_logvars = torch.cat(pred_logvars_parts, dim=1)
        eval_mse_losses, _ = self._model_loss(pred_means, pred_logvars, targets, mse_only=True)
        eval_mse_losses_np = eval_mse_losses.detach().cpu().numpy()
        if update_elite_models:
            elite_indices = np.argsort(eval_mse_losses_np)[:self._d_fns.num_elite]
            self._d_fns.set_elite_model_indices(elite_indices)
        return eval_mse_losses_np, None

    @torch.no_grad()
    def predict(
            self,
            obs: Union[np.ndarray, Tensor],
            act: Union[np.ndarray, Tensor],
            reward_penalty_coef: float = 1.0,
            deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Predict next states and penalized rewards for MOPO rollouts."""
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        act = torch.as_tensor(act, device=self._device, dtype=torch.float32)
        scaled_obs, scaled_act = self.transform_obs_action(obs, act)
        inputs = torch.cat([scaled_obs, scaled_act], dim=-1)
        pred_diff_means, pred_diff_logvars = self._d_fns.predict(inputs)
        pred_diff_stds = torch.exp(0.5 * pred_diff_logvars)
        pred_diff_samples = pred_diff_means
        if not deterministic:
            pred_diff_samples = pred_diff_samples + torch.randn_like(pred_diff_means) * pred_diff_stds

        elite_indices = self._d_fns.elite_model_idxes
        batch_size = obs.shape[0]
        model_idxes = elite_indices[torch.randint(len(elite_indices), (batch_size,), device=self._device)]
        batch_idxes = torch.arange(batch_size, device=self._device)
        pred_diff = pred_diff_samples[model_idxes, batch_idxes]
        next_obs = pred_diff[:, :self._state_dim] + obs
        rewards = pred_diff[:, -1]

        if self._terminal_fn is None:
            terminals = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        else:
            terminals = torch.as_tensor(
                self._terminal_fn(
                    np.zeros_like(act.detach().cpu().numpy()),
                    obs.detach().cpu().numpy(),
                    act.detach().cpu().numpy(),
                    next_obs.detach().cpu().numpy(),
                ),
                dtype=torch.float32,
                device=self._device,
            )

        penalty = torch.amax(torch.norm(pred_diff_stds, dim=-1), dim=0)
        penalized_rewards = rewards - reward_penalty_coef * penalty
        info = {
            'penalty': penalty,
            'raw_rewards': rewards,
        }
        return next_obs, penalized_rewards, terminals, info

    def dynamics_fns(
            self,
            s: Union[np.ndarray, Tensor],
            a: Union[np.ndarray, Tensor],
    ) -> Tuple[List[Tensor], Dict[str, List[Normal]]]:
        s = torch.as_tensor(s, device=self._device, dtype=torch.float32)
        a = torch.as_tensor(a, device=self._device, dtype=torch.float32)
        scaled_s, scaled_a = self.transform_obs_action(s, a)
        inputs = torch.cat([scaled_s, scaled_a], dim=-1)
        pred_diff_means, pred_diff_logvars = self._d_fns.predict(inputs)
        pred_means = pred_diff_means.clone()
        pred_means[:, :, :self._state_dim] = pred_means[:, :, :self._state_dim] + s.unsqueeze(0)
        pred_stds = torch.exp(0.5 * pred_diff_logvars)
        s_pred = [pred_means[idx] for idx in range(pred_means.shape[0])]
        s_dist = [
            Normal(loc=pred_means[idx], scale=pred_stds[idx])
            for idx in range(pred_means.shape[0])
        ]
        return s_pred, {'dist': s_dist}

    def save(self, ckpt_name: str) -> None:
        torch.save(self._dyna_module.state_dict(), ckpt_name + '.pth')

    def restore(self, ckpt_name: str) -> None:
        self._dyna_module.load_state_dict(
            torch.load(ckpt_name + '.pth', map_location=self._device, weights_only=True)
        )

    @property
    def finished(self) -> bool:
        """Whether dynamics training should stop early."""
        return self._finished

    @property
    def dyna_nets(self) -> nn.ModuleList:
        """Expose individual ensemble members for compatibility."""
        return self._d_fns.ensemble_models


class DynaModule(BaseDynaModule):
    """Container for MOPO dynamics modules."""

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._d_net = self._net_modules.d_net_factory().to(device)

    @property
    def d_net(self) -> EnsembleDynamicsModel:
        return self._d_net
