import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Space
from typing import List, Optional, Sequence, Type, Union

ModuleType = Type[nn.Module]
LOG_STD_MIN = -20
LOG_STD_MAX = 2


def miniblock(
        input_size: int,
        output_size: int = 0,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


class CriticNetwork(nn.Module):
    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self._device = device
        self._layers = []
        hidden_sizes = [self.observation_dim + self.action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, activation=nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], 1)]
        self._model = nn.Sequential(*self._layers)

    def forward(self, x, a):
        x = torch.as_tensor(x, device=self._device, dtype=torch.float32)
        a = torch.as_tensor(a, device=self._device, dtype=torch.float32)
        x = torch.cat([x, a], dim=-1)
        x = self._model(x)
        return x.view(-1)


class ActorNetwork(nn.Module):
    """Gaussian actor with explicit tanh-squash correction, matching MOPO more closely."""

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self._device = device
        self._eps = np.finfo(np.float32).eps.item()
        self._layers = []
        hidden_sizes = [self.observation_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, activation=nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], self.action_dim * 2)]
        self._model = nn.Sequential(*self._layers)
        self.register_buffer(
            "_action_mags",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "_action_means",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = torch.as_tensor(x, device=self._device, dtype=torch.float32)
        output = self._model(x)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        pre_tanh_action = normal.rsample()
        squashed_action = torch.tanh(pre_tanh_action)
        action = squashed_action * self._action_mags + self._action_means

        log_prob = normal.log_prob(pre_tanh_action)
        log_prob = log_prob - torch.log(self._action_mags * (1 - squashed_action.pow(2)) + self._eps)
        log_prob = log_prob.sum(dim=-1)

        mode_action = torch.tanh(mean) * self._action_mags + self._action_means
        return mode_action, action, log_prob
