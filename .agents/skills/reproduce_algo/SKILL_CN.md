---
name: reproduce-algo
description: 将外部 RL 算法复现并迁移到 D2C-XJTU。适用于用户要求复现某个算法、把 baseline 仓库或论文实现移植到 D2C、对照 `algorithms_reproduction/...` 与 `d2c/...` 的模块映射，或在写代码前先制定实现计划。尤其适合 MOPO 这类基于模型的离线强化学习方法。
---

# 将外部算法复现到 D2C

当任务是把外部实现的算法复现到当前仓库时，使用这个 skill。目标不是按目录原样复制，而是做一版忠实且符合 D2C 代码结构的原生集成。

## 必读仓库锚点

在规划或动手修改前，先阅读这些文件：

- [docs/tutorials/developer.rst](../../docs/tutorials/developer.rst)
- [d2c/models/base.py](../../d2c/models/base.py)
- [d2c/models/__init__.py](../../d2c/models/__init__.py)
- `d2c/models/...` 下最接近的现有 agent 实现
- `d2c/trainers/...` 下相关 trainer
- `d2c/envs/learned/...` 下相关 dynamics 和 learned env 代码
- [example/benchmark/config/model_config.json5](../../example/benchmark/config/model_config.json5)
- `example/benchmark/` 和 `test/` 下对应的 demo 与测试

然后检查源算法实现：

- 入口脚本，例如 `train.py`
- 核心算法文件
- dynamics 或 model 文件
- replay buffer 工具
- 任务配置文件
- termination / reward 辅助逻辑
- 只有在会影响训练语义时，才去看 evaluation 代码

## 核心原则

除非 D2C 完全没有合适抽象，否则不要把源仓库 1:1 镜像过来。先把源实现拆成四类：

- D2C 中可以直接复用的部分
- 只需要做一层轻量适配的部分
- 必须高保真复现的算法特有逻辑
- benchmark 胶水代码，例如 task config 和 termination function

最脆弱、最影响结果的部分要优先保证 fidelity。对基于模型的离线 RL 来说，通常包括 rollout schedule、不确定性惩罚、elite model 选择、holdout 训练逻辑，以及 termination 处理。

## 规划输出格式

当用户要求先出计划时，按下面顺序组织：

1. 用一小段话说明源算法真实的训练闭环。
2. 给出 `source -> D2C target -> action` 形式的模块映射表。
3. 给出按优先级排序的实现顺序。
4. 说明哪些部分可以复用。
5. 说明风险点和可能的 fidelity 缺口。

尽量给出明确文件路径，而不是抽象描述。

## 执行流程

1. 先确定算法属于哪个 family，以及应该放到 `d2c/models` 的哪个子目录。
2. 找到最接近的 D2C 参考实现，使用当前 D2C 仓库中的 SAC 算法作为基础策略，相关使用模块尽可能使用当前 SAC 算法的模块。
3. 先读源实现入口，重构出真实训练流程。
4. 列出所有会改变行为的源模块，而不是只看 import。
5. 将每个源模块映射到以下四类之一：
   - `agent`
   - `dynamics / env`
   - `trainer / schedule`
   - `config / demo / tests`
6. 判断 D2C 现有组件里哪些在行为上等价、可以直接复用。
7. 先实现算法特有增量，再补 registration、config、demo 和 tests。
8. 先做聚焦验证，不要一上来就跑完整 benchmark。

## MOPO 默认映射模板

当复现 `algorithms_reproduction/mopo_pytorch/mopo` 时，默认按下面的映射来规划：

- `algo/mopo.py` -> `d2c/models/model_based/mopo.py` -> 实现 MOPO 主编排逻辑，包括离线数据采样、model rollout、real/model 混合 batch、`real_ratio`、`rollout_freq`、`rollout_length` 和 model buffer 管理。
- `algo/sac.py` -> 优先复用 `d2c/models/model_free/sac.py` 与 `d2c/networks_and_utils_for_agent/sac_nets_utils.py` 的模式 -> 在 `MOPOAgent` 内实现基于混合 batch 的 offline SAC 更新。
- `models/transition_model.py` -> 在 `d2c/envs/learned/dynamics/` 下新增 MOPO dynamics wrapper -> 保留 holdout split、elite model selection、best snapshot 跟踪、不确定性惩罚，以及 rollout 时的预测接口。
- `models/ensemble_dynamics.py` -> 新增 ensemble dynamics 网络支持 -> 如果现有 `ProbDyna` 无法高保真表达 elite ensemble 训练与源代码的不确定性逻辑，就补一个 MOPO 专用 ensemble model。
- `static_fns/*.py` -> benchmark 专用 termination helper -> 这部分必须迁移，因为 D2C 默认 benchmark `done_fn` 不足以支撑 MOPO rollout。
- `common/buffer.py` -> 通常复用 `d2c/utils/replaybuffer.py` -> 只在需要时补 mixed sampling 和 batched model rollout insert helper，不要轻易整套重写 buffer。
- `trainer.py` -> `d2c/trainers/trainer.py` 或专用 trainer hook -> 支持 `train_schedule: ['d', 'agent']`，先训练 dynamics，再在 agent 训练阶段周期性 rollout。
- `config/*.py` -> `example/benchmark/config/model_config.json5` 与 `example/benchmark/demo_mopo.py` -> 将任务相关超参数接入 D2C config 和 demo 入口。

## MOPO 推荐实现顺序

1. 先迁移 Hopper、Walker2d、HalfCheetah 的 termination function。
2. 再实现 MOPO dynamics 和 elite-model 训练逻辑。
3. 然后实现带 model rollout 与 mixed-batch SAC update 的 `MOPOAgent`。
4. 再补 agent 注册、config 和 demo 接线。
5. 最后补单测和一次短 benchmark smoke run。

## Fidelity 检查清单

在声称“复现完成”之前，确认下面这些点：

- dynamics 是否先于 policy optimization 训练
- dynamics 的训练是否为 `algorithms_reproduction/mopo_pytorch/mopo/algo/mopo.py` 中的将数据集划分batch的迭代器形式训练
- dynamics 训练过程中是否成功的 update_best_snapshots
- 确认 dynamics 最长训练步数以及什么条件下会提前停止 dynamics 训练
- rollout transition 是否使用 learned-model termination，而不是默认全 0 done
- reward penalty 是否只被施加了一次
- real/model batch mixing 是否符合配置比例
- model buffer 的大小和 retention 逻辑是否有界
- target network update 和 entropy tuning 是否符合预期 schedule
- evaluation 是否跑在 external benchmark env 上，而不是 learned env
- config key 是否与 `model.model_name` 和注册表一致
- 执行 HalfCheetah_medium_replay-v2 的任务，监视 dynamics 训练过程中 dynamics loss 等数据不会爆炸，不能超过1000等
- 在完成 dynamics 训练后，保证 50k steps 内 SAC 的训练过程中，SAC 相关的 loss 等数据不会爆炸，不能超过1000等

## 常见坑

- 按外部仓库目录结构硬搬，而不是接入 D2C 现有抽象
- 明明会改变 MOPO 核心行为，却仍强行复用现有 D2C dynamics 模块
- 忘记迁移 `static_fns` 或同等 termination logic
- 把 rollout scheduling 只写在 demo 里，而不是训练主逻辑里
- 把源仓库里的 helper 细节默认当成“设计使然”，却没有说明取舍
- 写完核心算法文件就停下，没有补 registration、config、demo 和 tests

## 最小文件清单

通常至少会涉及这些文件：

- `d2c/models/model_based/mopo.py`
- `d2c/models/__init__.py`
- `d2c/envs/learned/dynamics/...`
- `example/benchmark/config/model_config.json5`
- `example/benchmark/demo_mopo.py`
- `test/models/.../test_mopo.py`
- 如果新增了 benchmark termination helper，对应文件也要补
- 如果算法正式纳入支持列表，还要更新 `README.md`

## 最终回复要求

使用这个 skill 时，最终回复应包含：

- 哪些部分复用了 D2C
- 哪些模块是新写的
- 具体创建或修改了哪些文件
- 跑了哪些验证
- 还剩哪些 fidelity 风险，尤其是 dynamics 训练和 rollout 行为相关的风险
