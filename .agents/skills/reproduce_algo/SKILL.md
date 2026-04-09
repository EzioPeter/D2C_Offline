---
name: reproduce-algo
description: Reproduce and port an external RL algorithm into D2C-XJTU. Use when the user asks to reproduce an algorithm, port a baseline repository or paper implementation into D2C, compare module mappings between `algorithms_reproduction/...` and `d2c/...`, or make an implementation plan before coding. Especially useful for model-based offline RL methods such as MOPO.
---

# Reproduce Algorithm Into D2C

Use this skill when the task is to reproduce an algorithm from an external implementation into this repository. The goal is not a directory-level copy. The goal is a faithful D2C-native integration.

## Required Repo Anchors

Read these before planning or editing:

- [docs/tutorials/developer.rst](../../docs/tutorials/developer.rst)
- [d2c/models/base.py](../../d2c/models/base.py)
- [d2c/models/__init__.py](../../d2c/models/__init__.py)
- The closest existing agent family under `d2c/models/...`
- The relevant trainer under `d2c/trainers/...`
- The relevant dynamics and learned env code under `d2c/envs/learned/...`
- [example/benchmark/config/model_config.json5](../../example/benchmark/config/model_config.json5)
- Matching demos and tests under `example/benchmark/` and `test/`

Then inspect the source implementation:

- The entry script such as `train.py`
- The core algorithm file
- Dynamics or model files
- Replay buffer utilities
- Task config files
- Termination or reward helpers
- Evaluation code only if it affects training semantics

## Core Rule

Do not mirror the source repo 1:1 unless D2C has no suitable abstraction. First separate the source into:

- D2C pieces you can reuse as-is
- Pieces that need thin adapters
- Algorithm-specific logic that must be implemented faithfully
- Benchmark glue such as task configs and termination functions

The fragile parts deserve the highest fidelity. In offline model-based RL this usually means rollout schedule, uncertainty penalty, elite model selection, holdout training logic, and termination handling.

## Planning Output

When the user asks for a plan, answer in this order:

1. One short paragraph describing the source algorithm's real training loop.
2. A module mapping list in the form `source -> D2C target -> action`.
3. A prioritized implementation order.
4. Reuse opportunities.
5. Risks or likely fidelity gaps.

Prefer concrete file paths over abstract descriptions.

## Execution Workflow

1. Identify the algorithm family and target folder in `d2c/models`.
2. Find the closest D2C reference implementation. Use the current SAC algorithm in the D2C repository as the base policy, and must reuse modules from the current SAC implementation. Make sure the mopo's sac alpha loss follows the same with the sac alpha loss in the D2C sac implementation. Do not use some function like `get_alpha()` to simplify the implementation. Just follow the same logic as the sac implementation in D2C
3. Read the source entrypoint and reconstruct the true training loop.
4. List every source module that changes behavior, not just imports.
5. Map each source module into one of four buckets:
   - `agent`
   - `dynamics / env`
   - `trainer / schedule`
   - `config / demo / tests`
6. Decide which existing D2C components are behaviorally equivalent and can be reused.
7. Implement the algorithm-specific delta first, then wire registration, config, demo, and tests.
8. Run focused verification instead of jumping straight to full benchmarks.

## MOPO Default Mapping

When reproducing `algorithms_reproduction/mopo_pytorch/mopo`, use this mapping as the default plan:

- `algo/mopo.py` -> `d2c/models/model_based/mopo.py` -> implement MOPO orchestration: offline dataset sampling, model rollout, mixed real/model batches, `real_ratio`, `rollout_freq`, `rollout_length`, and model buffer management.
- `algo/sac.py` -> reuse patterns from `d2c/models/model_free/sac.py` and `d2c/networks_and_utils_for_agent/sac_nets_utils.py` -> implement offline SAC updates on mixed batches inside `MOPOAgent`.
- `models/transition_model.py` -> new MOPO dynamics wrapper in `d2c/envs/learned/dynamics/` -> preserve holdout split, elite model selection, best snapshot tracking, uncertainty penalty, and rollout-time prediction API.
- `models/ensemble_dynamics.py` -> new ensemble dynamics network support -> add a MOPO-specific ensemble model if `ProbDyna` cannot express elite ensemble training and the source uncertainty logic faithfully.
- `static_fns/*.py` -> benchmark-specific termination helpers -> port these because D2C's default benchmark `done_fn` is not sufficient for MOPO rollouts.
- `common/buffer.py` -> usually reuse `d2c/utils/replaybuffer.py` -> only add helpers for mixed sampling and batched model rollout inserts; do not clone the whole buffer unless required.
- `trainer.py` -> `d2c/trainers/trainer.py` or a dedicated trainer hook -> support `train_schedule: ['d', 'agent']`, pretrain dynamics first, then periodic rollouts during agent training.
- `config/*.py` -> `example/benchmark/config/model_config.json5` and `example/benchmark/demo_mopo.py` -> move task-specific hyperparameters into D2C config and demo arguments.

## Recommended MOPO Order

1. Port task termination functions for Hopper, Walker2d, and HalfCheetah.
2. Implement MOPO dynamics and elite-model training.
3. Implement `MOPOAgent` with model rollout and mixed-batch SAC updates.
4. Register the agent and add config and demo wiring.
5. Add unit tests and a short benchmark smoke run.

## Fidelity Checks

Before claiming reproduction is complete, verify:

- check the mopo's sac alpha loss follows the same with the sac alpha loss in the D2C sac implementation. Do not use some function like `get_alpha()` to simplify the implementation. Just follow the same logic as the sac implementation in D2C
- dynamics are trained before policy optimization
- dynamics training follows the dataset-batch iterator style used in `algorithms_reproduction/mopo_pytorch/mopo/algo/mopo.py`
- `update_best_snapshots` succeeds during dynamics training
- the maximum number of dynamics training steps is defined clearly, and the early-stop conditions for dynamics training are implemented correctly
- rollout transitions use learned-model termination, not zero-done defaults
- reward penalty is applied exactly once
- real/model batch mixing matches the configured ratio
- model buffer size and retention logic are bounded
- target network updates and entropy tuning match the intended schedule
- evaluation uses the external benchmark environment, not the learned env
- config keys match `model.model_name` and registration entries
- run the `HalfCheetah_medium_replay-v2` task and monitor dynamics training to ensure dynamics losses and related statistics do not explode, for example exceeding 1000
- after dynamics training is complete, ensure that during the first 50k SAC training steps, SAC-related losses and statistics do not explode, for example exceeding 1000

## Common Pitfalls

- Copying the external repo directory layout instead of integrating into D2C abstractions
- Reusing an existing D2C dynamics module even when it changes core MOPO behavior
- Forgetting `static_fns` or equivalent termination logic
- Putting rollout scheduling only in the demo script instead of the actual training code
- Treating source helper quirks as intentional without noting the choice
- Stopping after the core algorithm file without registration, config, demo, and tests

## Minimal File Checklist

Expect to touch at least:

- `d2c/models/model_based/mopo.py`
- `d2c/models/__init__.py`
- `d2c/envs/learned/dynamics/...`
- `example/benchmark/config/model_config.json5`
- `example/benchmark/demo_mopo.py`
- `test/models/.../test_mopo.py`
- benchmark-specific termination helper files if introduced
- `README.md` if the algorithm becomes a supported method

## Final Response Contract

When using this skill, the final response should include:

- what was reused from D2C
- what new modules were required
- the exact files created or changed
- what verification ran
- the remaining fidelity risks, especially around dynamics training and rollout behavior
