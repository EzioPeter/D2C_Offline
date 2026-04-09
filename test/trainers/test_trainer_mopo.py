import os
import shutil
import pytest
import numpy as np
from d2c.data import Data
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import LeaEnv
from d2c.utils.utils import abs_file_path
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


class TestTrainerMopo:
    device = 'cpu'
    work_abs_dir = abs_file_path(__file__, '../../example/benchmark')
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
        'train.batch_size': 16,
        'train.total_train_steps': 4,
        'train.summary_freq': 1,
        'train.print_freq': 1,
        'train.save_freq': 4,
        'train.eval_freq': 4,
        'train.model_buffer_size': 256,
        'train.agent_ckpt_dir': './temp/mopo_trainer_hc/agent/agent',
        'train.dynamics_ckpt_dir': './temp/mopo_trainer_hc/dynamics/dynamics',
        'train.wandb.mode': 'disabled',
        'model.mopo.hyper_params.rollout_freq': 1,
        'model.mopo.hyper_params.rollout_batch_size': 32,
        'model.mopo.hyper_params.rollout_length': 1,
        'model.mopo.hyper_params.real_ratio': 0.5,
    }
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()
    lea_env = LeaEnv(config)

    def test_trainer_mopo(self):
        shutil.rmtree('./temp/mopo_trainer_hc', ignore_errors=True)
        train_data = get_train_data(self.config)
        agent = make_agent(config=self.config, env=self.lea_env, data=train_data)
        trainer = Trainer(
            agent=agent,
            train_data=train_data,
            config=self.config,
            env=self.lea_env,
            evaluator=None,
        )
        trainer.train()
        assert os.path.exists(self.config.model_config.train.dynamics_ckpt_dir + '.pth')
        assert os.path.exists(self.config.model_config.train.agent_ckpt_dir + '.pth')


if __name__ == '__main__':
    pytest.main(__file__)
