# Relax-SDPO 示例

本目录提供 GRPO/Relax-SDPO 的启动脚本，以及 SciKnowEval L3 和工具调用数据的离线准备
入口。Relax-SDPO 默认使用独立的静态 teacher checkpoint；也可以显式启用 EMA teacher。
EMA 不创建新的训练 actor，而是在已有 actor 内维护 `actor_ema` CPU shadow，并在每个
actor training call 后同步到 managed SGLang teacher。
EMA teacher 启动时先使用 `TEACHER_MODEL_PATH` 作为可启动的 bootstrap，随后会在首个
rollout 前接收实际 actor（包括 resume 后 actor）的完整 `actor_ema` 权重。

## 模型与数据

模型需要两个路径：

```text
xxx/Qwen3-4B-Instruct-2507        # student
xxx/Qwen3-4B-Instruct-2507        # teacher
```

训练数据优先使用参考实现 `lasgroup/SDPO` 已整理的 split：

```text
xxx/Research/SDPO/datasets/sciknoweval/chemistry/train.json
xxx/Research/SDPO/datasets/sciknoweval/physics/train.json
xxx/Research/SDPO/datasets/sciknoweval/biology/train.json
xxx/Research/SDPO/datasets/sciknoweval/material/train.json
xxx/Research/SDPO/datasets/tooluse/train.json
```

对应的 `test.json` 用于后续独立验证；当前示例启动脚本只消费 `train.jsonl`。四个 SciKnowEval domain 为 Chemistry、Physics、Biology
和 Materials（参考目录名为 `material`）。参考仓库的 `tooluse` 与 ToolAlpaca 是两套不同
的数据入口：

```text
xxx/Research/SDPO/datasets/tooluse/train.json
xxx/Research/SDPO/datasets/tooluse/test.json
xxx/Data/ToolAlpaca/data/train-00000-of-00001.parquet
xxx/Data/ToolAlpaca/data/test-00000-of-00001.parquet
```

数据来源：

- 参考数据格式与 feedback 设计：<https://github.com/lasgroup/SDPO>
- SciKnowEval：<https://github.com/HICAI-ZJU/SciKnowEval>
- ToolAlpaca：<https://huggingface.co/datasets/Ahren09/ToolAlpaca>

## 数据准备

从项目根目录执行，按 domain 和 split 重复运行 SciKnowEval：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --output xxx/Data/SDPO/sciknoweval/chemistry/train.jsonl
```

ToolAlpaca：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset toolalpaca \
  --input xxx/Data/ToolAlpaca/data/train-00000-of-00001.parquet \
  --source-split train \
  --output xxx/Data/SDPO/relax/toolalpaca/train.jsonl
```

如果使用参考仓库的工具调用数据，将 `--dataset` 改为 `tooluse`，并将输入替换为
`xxx/Research/SDPO/datasets/tooluse/train.json` 或 `test.json`。

点火训练可以加 `--max-rows 2` 生成小数据文件。输出文件至少需要包含 `prompt`、
`label` 和 `metadata` 字段。

## 启动训练

准备好模型和数据后设置路径并运行对应脚本：

```bash
export STUDENT_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507
export TEACHER_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507
export DATA_PATH=xxx/Data/SDPO/sciknoweval/chemistry/train.jsonl
export RELAX_PYTHON=xxx/venv/relax-sdpo/bin/python
export MEGATRON=xxx/venv/relax-sdpo-megatron
export PYTHONPATH="${MEGATRON}:xxx/Research/relax-worktree"

MODE=grpo bash examples/on_policy_distillation/sdpo/run-2gpu.sh
MODE=sdpo bash examples/on_policy_distillation/sdpo/run-2gpu.sh
```

启用 EMA teacher 时，在 SDPO 启动参数中增加：

```text
--sdpo-teacher-update-mode ema
--sdpo-teacher-ema-alpha 0.01
```

EMA 发布间隔固定为 1；当前版本尚未加入 optimizer overflow/skip-step/NaN 成功门控。
EMA 仅支持 managed、single-teacher、colocate、full-model 的文本 SDPO；外部 teacher URL、
MOPD routes、hybrid/fully-async 和 LoRA 会被参数校验拒绝。

同一个入口也可用 `REPO=main` 运行 main checkout；此时设置
`MAIN_REPO_ROOT`、`MAIN_RELAX_PYTHON` 和 `MAIN_MEGATRON`。

`relax-sdpo` 是 Python/PyTorch 环境，`relax-sdpo-megatron` 是与当前 worktree 配套的
Megatron 源码目录。

双卡点火训练可设置 `ACTOR_GPUS=2`、`TP_SIZE=2`、`CP_SIZE=1`。GRPO 默认使用 actor=2、
rollout=2；Relax-SDPO 默认使用 actor=2、rollout=1、teacher=1。还可以通过
`N_SAMPLES_PER_PROMPT`、`GLOBAL_BATCH_SIZE`、`NUM_ROLLOUT` 和 `LEARNING_RATE` 调整规模；
`OPD_TOP_K` 只对 Relax-SDPO 脚本生效。启动脚本使用当前 Ray 环境，不会自动停止或清理其他进程。
