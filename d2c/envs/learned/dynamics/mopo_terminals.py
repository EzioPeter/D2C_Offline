"""Task-specific terminal functions used by MOPO rollouts."""

import numpy as np
from typing import Callable


def halfcheetah_termination_fn(
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
) -> np.ndarray:
    """HalfCheetah has no early termination in the MOPO benchmark setting."""
    del act, next_obs
    return np.zeros(obs.shape[0], dtype=bool)


def hopper_termination_fn(
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
) -> np.ndarray:
    """Ported from the reference MOPO implementation."""
    del obs, act
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = np.isfinite(next_obs).all(axis=-1) \
        * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
        * (height > 0.7) \
        * (np.abs(angle) < 0.2)
    return ~not_done


def walker2d_termination_fn(
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
) -> np.ndarray:
    """Ported from the reference MOPO implementation."""
    del obs, act
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) \
        * (height < 2.0) \
        * (angle > -1.0) \
        * (angle < 1.0)
    return ~not_done


def _wrap_terminal_fn(fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> Callable:
    def done_fn(
            past_a: np.ndarray,
            s: np.ndarray,
            a: np.ndarray,
            next_s: np.ndarray,
    ) -> np.ndarray:
        del past_a
        return fn(s, a, next_s).astype('float32')
    return done_fn


def get_mopo_terminal_fn(env_name: str) -> Callable:
    """Return the benchmark-specific terminal function for MOPO."""
    domain = env_name.split('-')[0].lower()
    fn_map = {
        'halfcheetah': halfcheetah_termination_fn,
        'hopper': hopper_termination_fn,
        'walker2d': walker2d_termination_fn,
    }
    if domain not in fn_map:
        raise NotImplementedError(f'MOPO terminal function for {env_name} is not implemented.')
    return _wrap_terminal_fn(fn_map[domain])
