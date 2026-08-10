# Relax-SDPO 示例

本目录提供 Relax-SDPO 的文本训练示例。SDPO（Self-Distillation from
Preference Optimization）先让学生模型在自己的 rollout 上产生多个回答，再根据 reward
生成反馈或成功回答，并把这些信息加入 teacher prompt。teacher 使用加入反馈后的 prompt
重新计算 token-level log-probability，学生模型通过 on-policy distillation loss 学习如何修正
自己的回答。

当前示例使用 Relax 管理的 SGLang teacher，并以 colocate 方式在两张 GPU 上运行：训练时
actor 使用整个两卡资源，rollout 和 teacher 在 rollout 阶段分别使用一张 GPU。六个 launcher
都使用静态 teacher、文本输入、`student_topk` token selection 和 JSD loss。

> **运行前请先确认当前 pod 上没有其他 Ray 任务。** `env.sh` 会执行 `ray stop`，因此 source
> 它时会停止当前 pod 上的 Ray 进程。训练机上如果有其他任务正在运行，不要直接执行这些
> launcher。

## 训练流程

每个 launcher 的一次训练迭代大致经过以下阶段：

```text
prompt-data
    │
    ├── 学生 rollout：同一问题生成多个 response，组成一个 group
    │
    ├── custom reward：计算 score，并记录 feedback
    │
    ├── SDPO feedback：构造每个样本的 teacher prompt
    │       ├── SciKnowEval：共享同 group 内的成功回答，并加入当前样本反馈
    │       └── ToolUse：只加入当前样本的工具调用/格式反馈
    │
    ├── managed SGLang teacher：对动态 teacher prompt 计算 top-K log-probability
    │
    └── student_topk + JSD loss：更新学生模型
```

当前脚本使用以下关键配置：

| 配置 | 当前值 | 说明 |
| --- | --- | --- |
| `--use-opd` | 开启 | 启用 on-policy distillation |
| `--opd-type` | `sglang` | teacher 由 Relax 管理的 SGLang 服务提供 log-probability |
| `--opd-token-selection` | `student_topk` | 在学生 rollout 的 top-K token 集合上计算 SDPO 信号 |
| `--opd-log-prob-top-k` | `100` | 每个位置收集 100 个 token 的 log-probability |
| `--opd-kl-type` | `jsd` | 使用 JSD 形式的 token-level distillation criterion |
| `--opd-norm-mode` | `tail` | 保留 top-K 之外的 tail probability mass |
| `--opd-loss-coef` | `1.0` | 将 distillation signal 作为 loss 注入训练 |
| `--opd-kl-coef` | `0.0` | 不使用 advantage 形式的 OPD KL |
| `--opd-disable-rl-reward` | 开启 | 不把基础 RL outcome reward 注入 actor 优化；custom reward 仍用于 SDPO feedback |
| `--group-rm` | 开启 | 让同一个 prompt 的多个 rollout 进入同一 reward group |
| `--use-rollout-logprobs` | 开启 | 复用学生 rollout 阶段的 log-probability 数据 |
| `--colocate --offload` | 开启 | 在 rollout、teacher 和 actor 之间切换共享 GPU 资源 |

`student_topk` 模式需要 SGLang 支持按位置返回 token ID。launcher 会设置
`RELAX_OPD_PER_POS_TOKEN_IDS=1`；运行环境还必须安装对应的 SGLang source patch。详见
[通用 OPD 文档中的 SGLang Patch 说明](../README.md#sglang-patch)。

## Launcher 与资源布局

所有当前 launcher 都是单机两卡 colocate 配置：

```text
两张 GPU 的 colocate resource pool
├── actor   ：2 GPU，训练阶段使用整个 pool
├── rollout  ：1 GPU，rollout 阶段使用
└── teacher  ：1 GPU，rollout 阶段使用 managed SGLang teacher
```

脚本中的资源配置为：

```json
{"actor": [1, 2], "rollout": [1, 1], "teacher": [1, 1]}
```

| 脚本 | 数据入口 | 默认 rollout 配置 | Feedback 类 | teacher timeout |
| --- | --- | --- | --- | --- |
| [`run-sciknoweval-biology-2xgpu-colocate.sh`](run-sciknoweval-biology-2xgpu-colocate.sh) | `sciknoweval/biology/train.jsonl` | `num-rollout=50`，`n-samples-per-prompt=8`，`global-batch-size=32` | `SciKnowEvalSDPOFeedback` | 600 s |
| [`run-sciknoweval-chemistry-2xgpu-colocate.sh`](run-sciknoweval-chemistry-2xgpu-colocate.sh) | `sciknoweval/chemistry/train.jsonl` | `num-rollout=2`，`n-samples-per-prompt=2`，`global-batch-size=2` | `SciKnowEvalSDPOFeedback` | 120 s |
| [`run-sciknoweval-physics-2xgpu-colocate.sh`](run-sciknoweval-physics-2xgpu-colocate.sh) | `sciknoweval/physics/train.jsonl` | `num-rollout=2`，`n-samples-per-prompt=2`，`global-batch-size=2` | `SciKnowEvalSDPOFeedback` | 120 s |
| [`run-sciknoweval-material-2xgpu-colocate.sh`](run-sciknoweval-material-2xgpu-colocate.sh) | `sciknoweval/material/train.jsonl` | `num-rollout=2`，`n-samples-per-prompt=2`，`global-batch-size=2` | `SciKnowEvalSDPOFeedback` | 120 s |
| [`run-tooluse-2xgpu-colocate.sh`](run-tooluse-2xgpu-colocate.sh) | `tooluse/train.jsonl` | `num-rollout=2`，`n-samples-per-prompt=2`，`global-batch-size=2` | `ToolUseSDPOFeedback` | 120 s |
| [`run-toolalpaca-2xgpu-colocate.sh`](run-toolalpaca-2xgpu-colocate.sh) | `toolalpaca/train.jsonl` | `num-rollout=2`，`n-samples-per-prompt=2`，`global-batch-size=2` | `ToolUseSDPOFeedback` | 120 s |

其中 Chemistry、Physics、Materials、ToolUse 和 ToolAlpaca launcher 是两卡 smoke 配置；
Biology launcher 使用更大的默认 rollout 配置。当前脚本没有独立的公共 SDPO launcher，
每个数据入口都显式指定了自己的 feedback 类。

## 文件结构

| 文件 | 用途 |
| --- | --- |
| [`env.sh`](env.sh) | 设置项目根目录、Python/Megatron 环境、模型路径和数据根目录，并停止当前 Ray 进程 |
| [`prepare_data.py`](prepare_data.py) | 将参考数据转换为 Relax 的 `prompt`/`label`/`metadata` JSONL schema |
| [`reward.py`](reward.py) | 提供 SciKnowEval、ToolUse 和 ToolAlpaca 的 rule-based reward |
| `run-*-2xgpu-colocate.sh` | 按数据集启动两卡 colocate SDPO 训练 |

## 环境准备

### 模型

当前脚本通过 `scripts/models/qwen3-4B-Instruct-2507.sh` 加载 Qwen3-4B 的学生模型配置，
并默认使用同一 Qwen3-4B checkpoint 作为 student 和 teacher：

```text
<model-root>/Qwen3-4B-Instruct-2507  # student
<model-root>/Qwen3-4B-Instruct-2507  # teacher
```

也可以使用不同的文本 teacher checkpoint，但必须确保 checkpoint 能被当前 managed
SGLang teacher 和 Qwen3-4B 训练配置正确加载。当前 SDPO prompt-routing 路径只支持文本
输入，不支持多模态字段。

### `env.sh`

launcher 会自行回到项目根目录，并在内部 source `examples/on_policy_distillation/sdpo/env.sh`。
请先根据当前机器修改其中的环境路径和 checkpoint 路径：

| 变量 | 用途 |
| --- | --- |
| `RELAX_VENV` | Relax-SDPO Python 虚拟环境 |
| `RELAX_PYTHON` | 训练入口使用的 Python |
| `MEGATRON` | 与当前 Relax worktree 配套的 Megatron 和 Python package 路径 |
| `PYTHONPATH` | 由项目根目录和 `MEGATRON` 路径组成 |
| `STUDENT_MODEL_PATH` | 学生模型 HF checkpoint |
| `TEACHER_MODEL_PATH` | managed SGLang teacher 的 HF checkpoint |
| `SDPO_DATA_ROOT` | 准备好的 SDPO JSONL 数据根目录 |

`env.sh` 当前对上述变量使用固定默认值，而不是 `${VAR:-default}` 形式。因此，在命令行
预先 export `STUDENT_MODEL_PATH` 或 `TEACHER_MODEL_PATH` 会被 `env.sh` 中的赋值覆盖；如果
需要更换模型或运行环境，应直接修改 `env.sh`，或维护一份本地 launcher/environment 副本。

另外，`env.sh` 开头执行 `ray stop`。在训练机上执行前必须确认当前 pod 的 tmux 和 Ray
任务都为空；不要因为某台机器 GPU 空闲就假设它没有其他任务。

## 数据准备

从 Relax 项目根目录执行以下命令。输入可以是 JSON、JSONL 或 Parquet，输出统一为 JSONL。

### 输出 schema

每一行至少包含 `prompt`、`label` 和 `metadata`：

```json
{
  "prompt": "question and optional choices",
  "label": "gold answer",
  "metadata": {
    "data_source": "sciknoweval",
    "source_split": "train",
    "domain": "Chemistry",
    "task_type": "mcq"
  }
}
```

`metadata.data_source` 是 reward 路由键；`metadata` 中的 `answer_key`、`golden_answer` 等
字段由 reward 使用。不要在 launcher 中把 `--metadata-key metadata` 改成其他字段，除非
同时修改数据输出 schema 和 reward 逻辑。

### SciKnowEval

参考 `lasgroup/SDPO` 的数据通常按 domain 保存。以下命令以 Chemistry 为例；Physics、
Biology 和 Materials 只需要替换 domain、输入路径和输出路径：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sdpo-source-root>/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --output <data-root>/SDPO/sciknoweval/chemistry/train.jsonl
```

也可以处理已经整理成扁平 schema 的参考数据。对于原始 SciKnowEval 格式，转换器会只保留
L3 样本，并将 `material` 规范化为 `Materials`：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sciknoweval-root>/chemistry/train.json \
  --source-split train \
  --output <data-root>/SDPO/sciknoweval/chemistry/train.jsonl
```

测试 split 可以用同样的命令生成，只需将输入和 `--source-split` 改为 `test`。当前训练
launcher 默认只读取 `train.jsonl`，不会自动执行 test eval。

### ToolUse

如果使用参考 SDPO 仓库中的工具调用数据：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset tooluse \
  --input <sdpo-source-root>/datasets/tooluse/train.json \
  --source-split train \
  --output <data-root>/SDPO/tooluse/train.jsonl
```

### ToolAlpaca

ToolAlpaca 输入通常是 Parquet：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset toolalpaca \
  --input <toolalpaca-root>/data/train-00000-of-00001.parquet \
  --source-split train \
  --output <data-root>/SDPO/toolalpaca/train.jsonl
```

读取 Parquet 需要当前 Python 环境安装 `pyarrow`。

### Smoke 数据

点火训练前可以限制输出行数：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sdpo-source-root>/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --max-rows 2 \
  --output <data-root>/SDPO/sciknoweval/chemistry/train-smoke.jsonl
```

`--max-rows` 只截取转换后的样本，不改变数据 schema。使用 smoke 文件训练时，通过
`DATA_PATH` 覆盖 launcher 的默认数据路径。

## 启动训练

### 使用 `SDPO_DATA_ROOT` 默认路径

如果 `env.sh` 中的 `SDPO_DATA_ROOT` 已经包含以下目录之一，可以直接启动：

```text
$SDPO_DATA_ROOT/
├── sciknoweval/
│   ├── biology/train.jsonl
│   ├── chemistry/train.jsonl
│   ├── material/train.jsonl
│   └── physics/train.jsonl
├── toolalpaca/train.jsonl
└── tooluse/train.jsonl
```

例如：

```bash
bash examples/on_policy_distillation/sdpo/run-sciknoweval-chemistry-2xgpu-colocate.sh
```

### 覆盖数据路径和实验名

`DATA_PATH` 和 `EXPERIMENT_NAME` 在 launcher source `env.sh` 后读取，可以从命令行覆盖：

```bash
DATA_PATH=<data-root>/SDPO/sciknoweval/chemistry/train-smoke.jsonl \
EXPERIMENT_NAME=sdpo-sciknoweval-chemistry-smoke \
bash examples/on_policy_distillation/sdpo/run-sciknoweval-chemistry-2xgpu-colocate.sh
```

ToolAlpaca 和 ToolUse 的启动方式相同，只需替换 launcher：

```bash
DATA_PATH=<data-root>/SDPO/toolalpaca/train.jsonl \
bash examples/on_policy_distillation/sdpo/run-toolalpaca-2xgpu-colocate.sh

DATA_PATH=<data-root>/SDPO/tooluse/train.jsonl \
bash examples/on_policy_distillation/sdpo/run-tooluse-2xgpu-colocate.sh
```

训练开始前，Relax 会启动 managed SGLang teacher，并在 actor、rollout 和 teacher 之间按
colocate/offload 配置切换 GPU。脚本设置了 `--skip-eval-before-train`，所以启动后直接
进入训练，不会先读取 test 数据做 eval。

## Reward 与 Feedback

### SciKnowEval

`reward.py` 从回答中提取 `<answer>...</answer>` 内容；如果没有 answer tag，则从回答中
提取选项字母或使用完整回答进行比较。普通选择题会和 `metadata.answer_key` 比较，
true/false 任务会进行归一化比较。

失败样本得到类似以下的通用 feedback：

```text
The attempted answer is incorrect. Recheck the reasoning and final answer.
```

`SciKnowEvalSDPOFeedback` 会在同一 `group_index` 内寻找 `score >= 1` 的成功 response，
将其包装为 `<successful_attempt>`，并把当前样本的错误信息包装为 `<feedback>`。成功回答
本身没有 peer 时可以使用自己的回答；不同问题之间不会共享回答。

### ToolUse 与 ToolAlpaca

模型回答需要包含：

```text
Action: <tool name>
Action Input: <JSON object>
```

reward 会分别检查 tool action 和 JSON 参数；格式错误、tool 选择错误或参数不匹配都会
产生 score=0，并生成不泄露 gold answer 的错误反馈。`ToolUseSDPOFeedback` 对每个样本
单独构造 teacher prompt，不会把同 group 的其他工具调用答案注入当前样本。

## Teacher 模式与限制

当前六个 launcher 都显式使用：

```text
--sdpo-teacher-update-mode static
--sdpo-teacher-ema-alpha 0.01
```

实际运行模式是 `static`；因此 `--sdpo-teacher-ema-alpha` 在这些脚本中只是保留的参数值，
不会更新 teacher。框架支持 `ema` teacher，但它不是当前 launcher 的默认配置。手工改为
EMA 时还需要满足：

- 使用 Relax-managed 的单个 teacher checkpoint；
- 使用 `--colocate` 和 managed teacher resource；
- 使用 SGLang teacher、Megatron training backend 和 full-model training；
- 启用 `--enable-weights-backuper`；
- 不使用 MOPD routes、external teacher URL、hybrid、fully-async 或 LoRA；
- 继续使用 SDPO 所需的 `--group-rm`、`student_topk`、文本输入和 `--opd-loss-coef`。

如果只是想运行当前示例，请保持 `static`，不要只添加 `--sdpo-teacher-update-mode ema`。

## 常见问题

### `Set STUDENT_MODEL_PATH` 或 `Set SDPO_DATA_ROOT`

检查 `env.sh` 中的模型、Python/Megatron 和 `SDPO_DATA_ROOT` 配置。若使用自定义数据，
可以直接设置 `DATA_PATH`，这样 launcher 不需要依赖 `SDPO_DATA_ROOT` 对应的默认目录。

### `No rows matched`

检查 `--dataset` 是否与输入格式匹配。SciKnowEval 原始格式还需要有效的 L3 domain；
ToolAlpaca 输入必须包含 `golden_answer`；ToolUse 输入必须包含参考格式中的 `prompt` 和
`answer`。

### teacher 请求超时或显存不足

检查 teacher/rollout 是否确实各分配一张 GPU，并确认 `--colocate --offload` 没有被删除。
Biology launcher 的默认 rollout 规模明显大于其他脚本；首次验证建议使用其他 launcher
或先生成 `--max-rows 2` 的 smoke 数据。必要时应在对应 launcher 中调整 rollout 数量、
response 长度或 batch 配置。

### Top-K log-probability 不可用

确认 SGLang 已应用 [通用 OPD 文档中的 per-position token-id patch](../README.md#sglang-patch)，
并保留 `RELAX_OPD_PER_POS_TOKEN_IDS=1`。当前 SDPO 路径不能退回到
`student_sampled`，因为 SDPO prompt routing 只支持 `student_topk`。

## 参考

- [On-Policy Distillation 通用说明](../README.md)
- [通用 OPD 的 token selection、loss 和 SGLang 配置](../README.md#token-selection-modes)
- [lasgroup/SDPO](https://github.com/lasgroup/SDPO)
- [SciKnowEval](https://github.com/HICAI-ZJU/SciKnowEval)
- [ToolAlpaca](https://huggingface.co/datasets/Ahren09/ToolAlpaca)
