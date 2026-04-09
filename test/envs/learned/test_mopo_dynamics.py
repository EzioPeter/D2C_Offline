import os
import pytest
import numpy as np
from d2c.data import Data
from d2c.envs import LeaEnv
from d2c.envs.learned.dynamics import make_dynamics
from d2c.utils.utils import abs_file_path, maybe_makedirs
from d2c.utils.config import ConfigBuilder
from d2c.utils.replaybuffer import ReplayBuffer
from example.benchmark.config.app_config import app_config


def make_fake_data(config, size: int = 256) -> ReplayBuffer:
    state_dim = config.model_config.env.basic_info.state_dim
    action_dim = config.model_config.env.basic_info.action_dim
    device = config.model_config.train.device
    data = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=size,
        device=device,
    )
    s1 = np.random.randn(size, state_dim).astype(np.float32)
    a1 = np.random.uniform(-1.0, 1.0, size=(size, action_dim)).astype(np.float32)
    s2 = s1 + 0.1 * np.random.randn(size, state_dim).astype(np.float32)
    a2 = np.random.uniform(-1.0, 1.0, size=(size, action_dim)).astype(np.float32)
    r = np.random.randn(size).astype(np.float32)
    d = np.random.randint(0, 2, size=size).astype(np.float32)
    data.add_transitions(
        state=s1,
        action=a1,
        next_state=s2,
        next_action=a2,
        reward=r,
        done=d,
    )
    return data


def get_train_data(config) -> ReplayBuffer:
    dataset_path = config.model_config.env.external.data_file_path + '.hdf5'
    if os.path.exists(dataset_path):
        return Data(config).data
    return make_fake_data(config)


class TestMopoDynamics:
    device = 'cpu'
    work_abs_dir = abs_file_path(__file__, '../../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'HalfCheetah-v2',
        prefix + 'data_name': 'halfcheetah_medium_replay-v2',
        'model.model_name': 'mopo',
        'env.learned.dynamic_module_type': 'mopo',
        'env.learned.with_reward': True,
        'train.data_loader_name': None,
        'train.device': device,
        'train.batch_size': 32,
        'train.dynamics_ckpt_dir': './temp/mopo_env/dynamics/dynamics',
    }
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()

    def test_mopo_dynamics(self):
        maybe_makedirs(os.path.dirname(self.config.model_config.train.dynamics_ckpt_dir))
        train_data = get_train_data(self.config)
        dyna = make_dynamics(self.config, train_data)
        for _ in range(4):
            dyna.train_step()
        dyna.test_step()
        dyna.save(self.config.model_config.train.dynamics_ckpt_dir)

        batch = train_data.sample_batch(8)
        next_obs, rewards, dones, info = dyna.predict(batch['s1'], batch['a1'])
        state_dim = self.config.model_config.env.basic_info.state_dim
        assert next_obs.shape == (8, state_dim)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)
        assert info['penalty'].shape == (8,)

        env = LeaEnv(self.config)
        env.load()
        s_p, r, d, dist = env.step_raw(batch['s1'], batch['a1'], with_dist=True)
        model_num = self.config.model_config.env.learned.mopo.model_params.ensemble_size
        assert len(s_p) == len(r) == len(d) == len(dist) == model_num
        assert s_p[0].shape == (8, state_dim)
        assert r[0].shape == (8,)
        assert d[0].shape == (8,)
        assert dist[0].sample().shape == (8, state_dim + 1)


if __name__ == '__main__':
    pytest.main(__file__)
