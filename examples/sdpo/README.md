# Relax SDPO-lite：从数据准备到训练

这个目录提供 Relax 的SDPO 示例。

运行下面的命令前，请把示例中的 `xxx/...` 占位路径替换成当前机器上的实际路径。
本文档不依赖固定的目录布局。

当前实现的边界如下：

- student 和 teacher 是两个独立的模型实例；teacher 由 Relax 根据
  `--teacher-hf-checkpoint` 启动为独立的 SGLang 服务。
- teacher 使用静态 checkpoint，不接收梯度，不做权重同步，不做 EMA 或 trust-region
  更新。
- 训练目标是 `GRPO policy loss + masked differentiable OPD Top-K loss`。
- 只处理文本版 SciKnowEval L3 四个 domain 和参考 SDPO 仓库中的 `tooluse`；不需要
  code executor，也不实现 LiveCodeBench 训练/评测。
- 当前启动脚本显式设置 `--pipeline-model-parallel-size 1`，只考虑 TP/CP，不考虑 PP。
- 当前启动脚本是同步 `--colocate` 配置：actor placement group 为 4 GPU，rollout
  逻辑上占其中 2 GPU，teacher 逻辑上占其中 2 GPU，通过 offload/onload 复用同一组
  物理 GPU。默认物理资源需求是 4 GPU，不是 8 张独占 GPU。

## 1. 整体链路

训练时一组 prompt 会生成 `N_SAMPLES_PER_PROMPT` 个 response。reward 函数先对每个
response 产生分数和结构化反馈；随后 SDPO teacher-context router 只在同一 rollout
group 内寻找成功 response，并把成功 response/feedback 放到 teacher-only prompt 中：

```text
prepared JSONL
    │
    ▼
Relax rollout (student)
    │  N samples per prompt, group_index preserved
    ▼
examples.sdpo.reward.score
    │  score + feedback
    ▼
SDPO teacher-context routing
    │  feedback adapter → same-group selector → prompt renderer
    ▼
teacher prompt + original response
    │  static teacher SGLang prefill, Top-K log-probabilities
    ▼
GRPO loss + masked differentiable OPD loss
```

特权信息只存在于 teacher 输入中。student 仍然只对原始 prompt 和原始 response
计算 policy loss。teacher 输入的 token 布局是：

```text
student: [student prompt] + [original response]
teacher: [teacher prompt with solution/feedback] + [original response]
```

teacher prompt 会重新 tokenize，`teacher_prompt_length` 由重新 tokenize 的结果得到，
teacher prefill 实际使用 `max(teacher_prompt_length - 1, 0)` 作为
`logprob_start_len`。teacher 返回的第一个边界 log-prob 不参与蒸馏，随后
`response_length` 个 log-prob 对应原 response；因此成功解答和 feedback 引入的 token
偏移不会把 teacher 的 log-prob 对齐到错误的 response 位置。

## 2. 数据和模型来源

### 2.1 默认使用的参考 SDPO 数据

默认数据来自论文公开参考实现，而不是直接把 `xxx/Data/SciKnowEval` 下的
benchmark test 文件改名成 train：

- 上游仓库：[lasgroup/SDPO](https://github.com/lasgroup/SDPO)
- 本地 checkout：`xxx/Research/SDPO`
- 当前参考版本：`7c457fc1b1f636ae794eb0362ba37d4743b06fbc`
- 参考仓库的数据说明：
  `xxx/Research/SDPO/data/README.md`

参考仓库中的 SciKnowEval 派生数据来源于：

- 上游代码仓库：[HICAI-ZJU/SciKnowEval](https://github.com/HICAI-ZJU/sciknoweval)
- Hugging Face 数据集：[hicai-zju/SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval)

Relax 默认直接消费参考仓库已经完成 train/test 划分的 JSONL 文件：

| 逻辑数据集 | Relax 使用的 domain | train | test | 参考数据文件 |
| --- | --- | ---: | ---: | --- |
| SciKnowEval | Chemistry | 1890 | 210 | `datasets/sciknoweval/chemistry/{train,test}.json` |
| SciKnowEval | Physics | 720 | 80 | `datasets/sciknoweval/physics/{train,test}.json` |
| SciKnowEval | Biology | 450 | 50 | `datasets/sciknoweval/biology/{train,test}.json` |
| SciKnowEval | Materials | 841 | 94 | `datasets/sciknoweval/material/{train,test}.json` |
| Tool Use | tooluse | 4046 | 68 | `datasets/tooluse/{train,test}.json` |

上表中的文件名虽然是 `.json`，实际格式是 JSONL，即每行一个 JSON object。SciKnowEval
参考目录使用 `material`，Relax 输出目录统一使用 `materials`；不要手动修改源文件名
或 domain 字段。

SciKnowEval 四个 domain 的 train 总数为 3901，test 总数为 434；参考 `tooluse`
的 train/test 数量为 4046/68。

### 2.2 本地已有但不作为默认输入的数据

本地还存在 Hugging Face 的 SciKnowEval 数据快照：

```text
xxx/Data/SciKnowEval
```

当前其中的 `data/v1/sciknoweval_test_v1.jsonl` 和
`data/v2/sciknoweval_test_v2.jsonl` 是 benchmark test 数据，不包含本实验所需的
参考 SDPO train split，也不是完整的 SciKnowEval 评测代码仓库。因此不能只把它们
重命名为 `train.json` 后训练。

本地也存在原始 ToolAlpaca 数据：

- 上游：[Ahren09/ToolAlpaca](https://huggingface.co/datasets/Ahren09/ToolAlpaca)
- 本地路径：`xxx/Data/ToolAlpaca/data/train-00000-of-00001.parquet`
  和 `test-00000-of-00001.parquet`

参考 SDPO 的逻辑数据名是 `tooluse`；`Ahren09/ToolAlpaca` 是原始来源的另一种
导出格式。当前本地快照的 prompt 和 golden answer 与参考 `tooluse` 文件逐条等价，
但实验报告仍需记录实际使用的文件来源和 revision，不能把两份文件合并后重复统计。
默认指南使用参考仓库已经切分好的 `tooluse` 文件。

### 2.3 默认模型

当前启动脚本针对下面的 Qwen3-4B 配置：

```text
xxx/Qwen3-4B-Instruct-2507
```

student 和 teacher 使用 `xxx/Qwen3-4B-Instruct-2507` 这一 checkpoint；SDPO 模式下 Relax 会
分别加载两个模型服务，它们不是共享同一个可训练实例。可以通过
`STUDENT_MODEL_PATH` 和 `TEACHER_MODEL_PATH` 覆盖路径，但更换模型架构时还必须同步
检查 `examples/sdpo/launch_common.sh` 中的 Megatron model arguments，不能只替换
checkpoint 路径。

## 3. 数据准备

### 3.1 前置检查

在 Relax worktree 根目录执行。准备脚本会创建输出目录，但不会隐式划分 train/test，
也不会从一个文件随机切分数据；`--source-split` 是实验者明确指定的来源划分标签。

```bash
cd xxx/relax-worktree

test -d xxx/Research/SDPO
test -f xxx/Qwen3-4B-Instruct-2507/config.json
test -f xxx/Research/SDPO/datasets/sciknoweval/chemistry/train.json
test -f xxx/Research/SDPO/datasets/tooluse/train.json
```

如果参考仓库尚未 checkout，可以使用下面的来源命令；已有 checkout 时不要重复下载：

```bash
git clone https://github.com/lasgroup/SDPO.git \
  xxx/Research/SDPO
git -C xxx/Research/SDPO rev-parse HEAD
```

### 3.2 准备 SciKnowEval train/test

下面的八条命令分别处理四个 domain 的 train/test。输入路径中的 `material` 是参考仓库的真实目录
名，输出路径使用更清晰的 `materials`：

```bash
python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --output xxx/Data/SDPO/relax/sciknoweval/chemistry/train.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/chemistry/test.json \
  --domain chemistry \
  --source-split test \
  --output xxx/Data/SDPO/relax/sciknoweval/chemistry/test.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/physics/train.json \
  --domain physics \
  --source-split train \
  --output xxx/Data/SDPO/relax/sciknoweval/physics/train.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/physics/test.json \
  --domain physics \
  --source-split test \
  --output xxx/Data/SDPO/relax/sciknoweval/physics/test.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/biology/train.json \
  --domain biology \
  --source-split train \
  --output xxx/Data/SDPO/relax/sciknoweval/biology/train.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/biology/test.json \
  --domain biology \
  --source-split test \
  --output xxx/Data/SDPO/relax/sciknoweval/biology/test.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/material/train.json \
  --domain material \
  --source-split train \
  --output xxx/Data/SDPO/relax/sciknoweval/materials/train.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset sciknoweval \
  --input xxx/Research/SDPO/datasets/sciknoweval/material/test.json \
  --domain material \
  --source-split test \
  --output xxx/Data/SDPO/relax/sciknoweval/materials/test.jsonl
```

### 3.3 准备参考 Tool Use train/test

参考 SDPO 数据集名称是 `tooluse`，不是 `toolalpaca`：

```bash
python3 -m examples.sdpo.prepare_data \
  --dataset tooluse \
  --input xxx/Research/SDPO/datasets/tooluse/train.json \
  --source-split train \
  --output xxx/Data/SDPO/relax/tooluse/train.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset tooluse \
  --input xxx/Research/SDPO/datasets/tooluse/test.json \
  --source-split test \
  --output xxx/Data/SDPO/relax/tooluse/test.jsonl
```

这条链路不调用 HTTP API，也不运行工具。Tool Use reward 只比较模型输出中的
`Action` 和 `Action Input` 与 metadata 中的 golden answer；因此它适合当前的静态
文本训练 smoke path，但不是带真实环境执行结果的 agent 训练。

### 3.4 可选：准备 Ahren09/ToolAlpaca

只有在明确要做另一组 ToolAlpaca 实验时才使用下面的命令：

```bash
python3 -m examples.sdpo.prepare_data \
  --dataset toolalpaca \
  --input xxx/Data/ToolAlpaca/data/train-00000-of-00001.parquet \
  --source-split train \
  --output xxx/Data/SDPO/relax/toolalpaca/train.jsonl

python3 -m examples.sdpo.prepare_data \
  --dataset toolalpaca \
  --input xxx/Data/ToolAlpaca/data/test-00000-of-00001.parquet \
  --source-split test \
  --output xxx/Data/SDPO/relax/toolalpaca/test.jsonl
```

Parquet 输入需要当前环境安装 `pyarrow`。这组数据的 provenance、统计和实验名称
必须与参考 `tooluse` 分开记录。

## 4. 准备结果校验

先检查输出行数是否保持不变：

```bash
wc -l \
  xxx/Data/SDPO/relax/sciknoweval/chemistry/train.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/chemistry/test.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/physics/train.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/physics/test.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/biology/train.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/biology/test.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/materials/train.jsonl \
  xxx/Data/SDPO/relax/sciknoweval/materials/test.jsonl \
  xxx/Data/SDPO/relax/tooluse/train.jsonl \
  xxx/Data/SDPO/relax/tooluse/test.jsonl
```

期望行数为：Chemistry `1890/210`、Physics `720/80`、Biology `450/50`、Materials
`841/94`、Tool Use `4046/68`。若行数为 0，优先检查 `--dataset`、SciKnowEval
的 `--domain` 和输入文件的实际 schema。

每条输出记录至少包含以下字段：

```json
{
  "prompt": "student-side prompt",
  "label": "expected answer",
  "metadata": {
    "sdpo": true,
    "data_source": "sciknoweval or tooluse",
    "source_split": "train or test",
    "sdpo_prompt": "original prompt",
    "answer_key": "A",
    "golden_answer": "tooluse answer"
  }
}
```

其中 `answer_key` 是 SciKnowEval 的答案，`golden_answer` 是 Tool Use 的结构化
工具调用答案；它们只用于 reward，不会被拼进模型的 student prompt。`source_split`
用于在训练日志和后续评测中检查 train/test 泄漏。

## 5. 启动训练

### 5.0 训练运行环境

数据准备只依赖 Python 标准库（Parquet 可选依赖 `pyarrow`）；真正训练还需要已经
配置好的 Relax、Megatron、Ray、PyTorch 和 SGLang runtime。当前 worktree 没有独立
运行环境，后续合并到有 runtime 的 main dir 或训练 pod 后再执行本节命令。

当前是 Top-K OPD，SGLang 必须包含 per-position token-id log-prob patch：

```text
docker/patch/latest/sglang_per_pos_topk.patch
```

`launch_common.sh` 会自动设置并传播 `RELAX_OPD_PER_POS_TOKEN_IDS=1`，但不会替运行者
安装 SGLang patch。若当前 SGLang 尚未包含该 patch，需要在对应 SGLang source tree
中先应用它；没有 patch 时不要把 teacher Top-K 接口的启动失败误判为数据问题。

训练命令不会替你激活 Python 环境。执行前需要进入已经安装 Relax、`transfer_queue`、
Megatron、Ray、PyTorch 和 SGLang 的环境，并确认下面的导入检查通过：

```bash
python3 -c "import ray, torch, transfer_queue"
```

示例 `launch_common.sh` 会先为空值初始化这两个变量，以避免干净 shell 在 `set -u`
下提前退出；它不会替运行者猜测网卡。若当前 pod 需要指定
`NCCL_NVLS_ENABLE`/`NCCL_SOCKET_IFNAME`，优先使用训练环境已经提供的配置，不要在
共享训练 pod 上随意填写网卡名，或改走已建立 Ray 集群的 entrypoint 流程。

### 5.1 资源和默认参数

`examples/sdpo/launch_common.sh` 当前关键默认值如下；表中的 actor/rollout/teacher
GPU 是 colocate placement group 内的逻辑切分，不是需要同时占用的独立物理 GPU 数：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `STUDENT_MODEL_PATH` | `xxx/Qwen3-4B-Instruct-2507` | student checkpoint |
| `TEACHER_MODEL_PATH` | student path | 独立静态 teacher checkpoint |
| `ACTOR_GPUS` | `4` | colocate actor placement group 的 GPU 总数 |
| `ROLLOUT_GPUS` | `2` | rollout 在共享 placement group 中的逻辑 GPU 数 |
| `TEACHER_GPUS` | `2` | teacher 在共享 placement group 中的逻辑 GPU 数；只在 SDPO 模式使用 |
| `TP_SIZE` | `2` | student Megatron TP |
| `CP_SIZE` | `1` | student context parallel |
| `N_SAMPLES_PER_PROMPT` | `8` | GRPO group size，也是同组成功 response 路由的边界 |
| `GLOBAL_BATCH_SIZE` | `32` | rollout/training global batch |
| `OPD_TOP_K` | `100` | Top-K 近似蒸馏大小；SDPO 固定由 Student 选择 Top-K |
| `OPD_KL_TYPE` | `jsd` | Top-K KL/JSD 类型；§3 默认使用标准对称 JSD |
| `OPD_LOSS_COEF` | `1.0` | SDPO additive OPD loss 系数 |
| `NUM_ROLLOUT` | `200` | rollout iteration 数 |
| `ROLLOUT_BATCH_SIZE` | `4` | rollout batch size |
| `ROLLOUT_MAX_PROMPT_LEN` | `2048` | rollout prompt 最大 token 数 |
| `ROLLOUT_MAX_RESPONSE_LEN` | `2048` | rollout response 最大 token 数 |
| `LEARNING_RATE` | `1e-6` | 常数学习率 |
| `MAX_TOKENS_PER_GPU` | `4096` | dynamic batch 的 token 上限 |
| `ROLLOUT_ENGINE_GPUS` | `2` | 单个 rollout engine 的逻辑 TP GPU 数 |
| `SGLANG_MEM_FRACTION` | `0.8` | SGLang static memory fraction |
| `OPD_TEACHER_TIMEOUT_S` | `120` | teacher prefill 请求超时 |

SDPO 和 GRPO 的当前脚本都在 colocate 模式运行：SDPO 默认使用 4 张物理 GPU，
GRPO 也使用 actor/rollout 共享的 4-GPU placement group。若直接在 4-GPU pod 上由
`scripts/entrypoint/local.sh` 启动 Ray，需要显式设置 `NUM_GPUS=4` 或正确设置
`CUDA_VISIBLE_DEVICES`，因为 local entrypoint 在两者都未设置时默认向 Ray 宣称 8 GPU。
`TP_SIZE=1` 只改变 student TP，不会把 4-GPU placement group 自动变成更小的资源配置。

### 5.2 GRPO baseline

下面命令只启动 student rollout 和 actor，不使用 teacher，也不计算 OPD：

```bash
DATA_PATH=xxx/Data/SDPO/relax/sciknoweval/chemistry/train.jsonl \
STUDENT_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507 \
EXPERIMENT_NAME=sdpo-lite-grpo-sciknow-chemistry \
bash examples/sdpo/run-grpo.sh
```

替换 `DATA_PATH` 即可训练其它 domain，例如：

```bash
DATA_PATH=xxx/Data/SDPO/relax/sciknoweval/physics/train.jsonl \
EXPERIMENT_NAME=sdpo-lite-grpo-sciknow-physics \
bash examples/sdpo/run-grpo.sh
```

### 5.3 静态 teacher SDPO-lite

SDPO 模式额外启动一个独立 teacher engine：

```bash
DATA_PATH=xxx/Data/SDPO/relax/sciknoweval/chemistry/train.jsonl \
STUDENT_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507 \
TEACHER_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507 \
EXPERIMENT_NAME=sdpo-lite-sciknow-chemistry \
bash examples/sdpo/run-sdpo.sh
```

Tool Use 的命令只需替换数据路径和实验名：

```bash
DATA_PATH=xxx/Data/SDPO/relax/tooluse/train.jsonl \
STUDENT_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507 \
TEACHER_MODEL_PATH=xxx/Qwen3-4B-Instruct-2507 \
EXPERIMENT_NAME=sdpo-lite-tooluse \
bash examples/sdpo/run-sdpo.sh
```

脚本中的关键 Relax 参数是：

```text
--use-opd
--opd-type sglang
--teacher-hf-checkpoint <teacher checkpoint>
--opd-teacher-prompt-key prompt
--opd-token-selection student_topk
--opd-log-prob-top-k 100
--opd-loss-coef 1.0
--opd-kl-coef 0.0
--pipeline-model-parallel-size 1
--context-parallel-size <CP_SIZE>
```

`--opd-teacher-prompt-key prompt` 使现有 OPSD/OPD teacher prefill 通路生效；SDPO
prompt builder 在 reward 之后直接填充 `sample.teacher_prompt`，而不是改写训练数据
中的 student `prompt`。

### 5.4 切换 KL/JSD 类型

当前代码中的 JSD 是闭式计算，不需要从 teacher 分布采样第二次。当
`alpha=0.5` 时，对固定的 Student Top-K 词表以及 `tail` 模式追加的尾部概率计算：

\[
\operatorname{JSD}(P\|Q)=
\frac{1}{2}\operatorname{KL}(P\|M)+
\frac{1}{2}\operatorname{KL}(Q\|M),
\qquad M=\frac{P+Q}{2}.
\]

SDPO 固定使用 `student_topk`：student rollout 先提供每个 response position 的
Top-K token id，teacher 在增强 teacher prompt 下只对这些 token 重打分；训练时
Megatron student forward 会重新对同一组 token id 计算可微 student log-prob，
不会把 rollout 时保存的 student log-prob 直接当作梯度路径。SDPO 样本如果配置成
其它 token selection，Relax 会直接报错，避免实验静默退化成另一种算法。

标准对称 JSD 在 SDPO 示例中固定使用 `alpha=0.5`，并且是默认的 `OPD_KL_TYPE`。
如需做论文中的 KL 消融，可以显式设置 `OPD_KL_TYPE=reverse_kl` 或
`OPD_KL_TYPE=forward_kl`。Relax 还支持带权广义 JSD：

\[
M_\alpha=(1-\alpha)P+\alpha Q,\qquad
D_\alpha=(1-\alpha)\operatorname{KL}(P\|M_\alpha)+
\alpha\operatorname{KL}(Q\|M_\alpha).
\]

参考 SDPO 的 alpha 约定是 `alpha=0` 为 forward-KL、`alpha=1` 为 reverse-KL，只有
`0.5` 是上面的对称公式。Relax 的显式 `OPD_KL_TYPE=forward_kl/reverse_kl` 也遵循
这个分布方向。固定-K 的 `tail` bin 会参与 JSD；`OPD_NORM_MODE` 当前没有环境变量
覆盖入口，启动脚本固定传 `tail`。

下面是固定-K `student_topk + JSD` 的启动示例：

```bash
DATA_PATH=xxx/Data/SDPO/relax/sciknoweval/chemistry/train.jsonl \
OPD_KL_TYPE=jsd \
EXPERIMENT_NAME=sdpo-lite-jsd-sciknow-chemistry \
bash examples/sdpo/run-sdpo.sh
```

`sdpo_valid=False` 不会减少 teacher 请求次数：当前 `prefill()` 仍会为有 response
的 sample 发起 teacher prefill；它只在 loss 阶段屏蔽该 sample 的 OPD 数值/梯度贡献。

## 6. reward 和反馈路由规则

reward 函数位于 [`reward.py`](reward.py)，由
`--custom-rm-path examples.sdpo.reward.score` 加载，并通过 `--group-rm` 接收一组
样本。返回值包含 `score` 和 `feedback`，`--reward-key score` 用于 GRPO 归一化。

### SciKnowEval

- 从 response 的 `<answer>...</answer>` 或文本中提取 A/B/C/D；
- 与 `metadata.answer_key` 比较；
- 正确 response 的 `score` 为 `1.0`；错误 response 的 `score` 为 `0.0`，并返回
  修正提示。

### Tool Use

- 解析 `Action:` 和 `Action Input:`；
- 比较 tool name 和 JSON 参数；
- 不发起 HTTP 请求，不运行数据里的工具；
- 格式错误或参数错误会写入 feedback，并把 score 置为 `0.0`。

### SDPO sample routing

`relax/utils/opd/sdpo/prompt_builder.py` 采用小型组合式结构。默认入口
`prepare_sdpo_teacher_prompts()` 创建一个 `SdpoPromptBuilder`，由它完成 reward
反馈读取、同组路由、成功 response 选择和 SDPO 字段写回；`TeacherPromptRenderer` 只负责
渲染 teacher-only prompt，`FeedbackRecord` 负责保存标准化反馈，`SdpoPromptStats` 负责
输出路由统计。

当前默认 provider 直接读取 Relax 的 scalar/dict reward，避免为只有一种反馈来源的
SDPO-lite 引入多层策略类。真实环境接入时，可以通过 `SdpoPromptBuilder` 的
`feedback_provider(sample) -> FeedbackRecord` 注入 executor/API 反馈；group 路由、prompt
模板和 OPD token offset 无需改动。当前示例仍不启动工具环境。

路由规则是：

1. 只在相同 `group_index` 的 rollout samples 中选择成功 response；
2. 默认 `score >= 1.0` 才是成功 demonstration；
3. 默认不把当前 sample 自己当作自己的 demonstration；
4. 从 reward dict 的 `feedback`、`feedback_raw` 或 `error` 中读取反馈；
5. 有成功 response 或可用 feedback 时设置 `sample.sdpo_valid=True`；
6. 没有有效 teacher context 时仅屏蔽 SDPO/OPD 项，GRPO policy loss 仍然保留。

这保证了不同 prompt 的成功解答不会因为 batch 拼接而互相泄露。`group_index` 缺失
时，每个 sample 被当作 singleton，不会跨 sample 借用成功解答。`sdpo_valid` 是
sample-level mask，不是 token-level feedback mask；OPD 聚合会把它同时应用到 numerator
和有效 token denominator，因此只对有效 teacher context 的 response token 归一化。GRPO
policy、entropy 和 reference-KL 使用自己的原始 loss mask，不会被这个 sample-level mask
改变。全 batch 没有有效 sample 时，OPD 项退化为可反向传播的零值。

## 7. TP、CP、padding 和 micro-batch 注意事项

- student Megatron 使用 `TP_SIZE`；teacher SGLang 使用
  `TEACHER_GPUS`/`--teacher-num-gpus-per-engine` 的独立 TP 布局。teacher 是独立的
  Ray/SGLang 服务和模型实例，但在当前 `--colocate` 脚本中与 actor/rollout 共享同一
  placement group，通过 bundle 切分和 offload/onload 复用物理 GPU。
- `CP_SIZE=1` 是当前最稳妥的起点。脚本始终传入 `--calculate-per-token-loss`，因此
  解除了 Megatron-Bridge 对 `CP_SIZE>1` 的启动前置条件；普通 zig-zag CP 会按同一
  response offset 对齐 response log-probs、Student Top-K ids、student log-probs 和
  teacher log-probs。默认 4 张 actor GPU、`TP_SIZE=2` 时，`CP_SIZE=2` 才满足并行
  world-size 整除约束，仍需在有 GPU 的环境中验证。
- `allgather_cp=True` 与 Top-K OPD 当前显式不兼容，会抛出
  `NotImplementedError`，避免把全局连续布局静默当作 zig-zag 布局。
- padding token 不进入有效 response token 的 KL 统计；teacher prompt 新增的特权
  token 也不进入 student loss。
- 动态 batch/micro-batch 切分只改变传输分块，不应改变有效 response token 的
  `student_topk_token_ids`、teacher log-probs 和 `sdpo_valid` 对齐关系；这部分需要
  在有 GPU 的环境中用 TP/CP 集成测试确认，当前 worktree 尚未完成真机验收。

## 8. 运行位置和安全要求

`run-grpo.sh` 和 `run-sdpo.sh` 会进入 worktree 根目录。没有外部 entrypoint 时，
`launch_common.sh` 会 source `scripts/entrypoint/local.sh`；该 local entrypoint 会
尝试清理本机 stale Python/SGLang/Ray 进程并启动单机 Ray。因此：

- 只在确认没有其它任务的开发 pod 或专用单机上直接执行上面的 `bash` 命令；
- 在训练 pod 或共享 Ray 集群上，先确认 Ray、GPU 和 tmux 中没有其它任务，再使用项目
  的 `scripts/entrypoint/ray-job.sh`/集群提交流程；不要因为本示例而停止其它用户的
  Ray job 或进程；
- 当前 worktree 没有独立运行环境，不能在这里把“命令已写好”当作训练已经成功。

如果使用已经建立好的 Ray 集群，入口脚本至少应从包含当前 worktree 的环境执行，
并保证 Ray actor 能看到下面这些共享路径：

```text
xxx/relax-worktree
xxx/Data/SDPO/relax
xxx/Qwen3-4B-Instruct-2507
```

## 9. 单测、冒烟检查和当前验收边界

在具备 Relax 依赖的环境中，先做不启动 GPU 的静态检查：

```bash
cd xxx/relax-worktree

bash -n examples/sdpo/launch_common.sh examples/sdpo/run-grpo.sh examples/sdpo/run-sdpo.sh

python3 -m compileall -q \
  examples/sdpo \
  relax/utils/opd \
  relax/engine/rollout/on_policy_distillation.py \
  relax/backends/megatron/loss.py \
  relax/backends/megatron/data.py

PYTHONPATH=. pytest -q \
  tests/utils/opd/test_sdpo_prompt_builder.py \
  tests/utils/opd/test_sdpo_teacher_offset.py \
  tests/examples/sdpo/test_prepare_data.py \
  tests/examples/sdpo/test_data_reward.py \
  tests/backends/megatron/test_opd_loss_aggregation.py
```

当前这组 CPU 单测守护以下局部契约：JSON/JSONL 数据读取、same-group feedback/成功
解答路由、self-success 排除、Tool Use reward 解析、teacher response suffix/offset、
SDPO 强制使用 Student Top-K、sample-level `sdpo_valid` mask、通用 OPD token-mean
聚合，以及 JSD tail bin 的数值行为。它们不等价于完整的 teacher tokenizer/prefill、Top-K policy loss 或多卡
验收；完整 GRPO+SDPO loss 仍需要后续补充集成测试。

当前尚未在这个 worktree 声称完成的内容包括：

- Ray/SGLang teacher 的真实启动和 HTTP prefill；
- Qwen3-4B 的端到端 rollout → reward → dynamic prompt → teacher Top-K → Megatron
  loss；
- TP/CP 多卡、padding、dynamic CP 和 micro-batch 的真机统计等价性；
- 吞吐、峰值显存、teacher 额外 forward 时间和最终效果；
- 论文要求的 EMA teacher、LiveCodeBench、三 seed 结果和完整消融。

因此，本 README 的命令是“准备后可提交到有环境的机器执行”的最小实现指南，不能
替代后续的训练验收报告。

## 10. 相关实现文件

- 数据入口：[`prepare_data.py`](prepare_data.py)
- 可复用数据 normalizer/strategy：[`data_normalizers.py`](data_normalizers.py)
- SciKnowEval/Tool Use reward：[`reward.py`](reward.py)
- 统一启动参数：[`launch_common.sh`](launch_common.sh)
- GRPO baseline：[`run-grpo.sh`](run-grpo.sh)
- 静态 teacher SDPO-lite：[`run-sdpo.sh`](run-sdpo.sh)
- SDPO public API：[`relax/utils/opd/sdpo/__init__.py`](../../relax/utils/opd/sdpo/__init__.py)；
  prompt builder implementation：
  [`relax/utils/opd/sdpo/prompt_builder.py`](../../relax/utils/opd/sdpo/prompt_builder.py)
- 动态 teacher prefill：[`relax/engine/rollout/on_policy_distillation.py`](../../relax/engine/rollout/on_policy_distillation.py)
- teacher input/token offset：[`relax/utils/opd/opd_opsd_worker.py`](../../relax/utils/opd/opd_opsd_worker.py)
