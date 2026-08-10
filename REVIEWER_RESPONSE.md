# Reviewer Response

感谢建议。实现已收敛到现有 OPD/OPSD 主链路，新增逻辑集中在 reward 后的 dynamic teacher
prompt、sample-level `opd_sample_mask` 和可选 EMA teacher update。

代码注入点位于 `relax/engine/rollout/sglang_rollout.py` 的
`generate_and_rm_group()`，具体在 `batched_async_rm()` 完成后、
`state.opd_manager.prefill()` 发起前：

```text
group rollout
  → batched_async_rm(args, group)
  → EnvironmentFeedback.record_sample_feedback()
  → EnvironmentFeedback.prepare_teacher_prompts(group, rewards)
  → OpdManager.prefill(group)
  → existing teacher transfer / OPD loss
```

两个接口都是统一调用路径，由分配到的 feedback 类提供实现：`OPDFeedback` 和 `OPSDFeedback`
的 `record_sample_feedback()` 为空；`SciKnowEvalSDPOFeedback`、`ToolUseSDPOFeedback`、
`CodeSDPOFeedback` 记录各自 sample 的 feedback。随后 `prepare_teacher_prompts()` 接收完整
group 和对应 rewards：三个 SDPO 类都优先按 `group_index`（没有时按 metadata 中的 `uid`）选择
同组成功 peer，没有 peer 时允许成功样本自引用，绝不跨组/UID；当前 sample 的 feedback 仍只注入
当前 sample。plain OPD 使用 rollout tokens，ordinary OPSD 使用数据侧的 `Sample.teacher_prompt`，
SDPO 才在此处写入动态 `Sample.teacher_prompt` 和 `opd_sample_mask`；student prompt、response、
rollout token ids 和 student Top-K ids 不变。
teacher 仍对原 response 在 student-selected Top-K support 上重打分。

没有 solution/feedback 时 teacher prompt 回退到原 prompt，但 teacher 并不实时跟随 student，
因此该 sample 的合法 teacher log-prob 仍可能产生错误的非零 OPD 梯度；此时
`opd_sample_mask=False`，在 OPD loss reduction 中屏蔽该 sample，基础 RL loss 不受影响。普通
OPD/OPSD 未携带 `opd_sample_mask` 时仍沿用 upstream/main 的原始 OPD loss reducer 和归一化。

数学公式也沿用 upstream/main：JSD `alpha=0` 是 `KL(student || teacher)`，`alpha=1` 是
`KL(teacher || student)`；SDPO 不反转这两个端点。

teacher update 支持 `static` 和 `ema`：`static` 保持初始 teacher，`ema` 在 actor update
后按 `--sdpo-teacher-ema-alpha` 更新并发布 teacher snapshot，alpha 要求大于 0 且不超过 1，默认
`0.01`。EMA 仅适用于单个 colocated managed SGLang teacher、Megatron 全量训练和
`--enable-weights-backuper` 配置。

下一步重点是在单个数据集上完成几十步真实训练，验证 reward→prompt→prefill→Top-K
rescoring→mask reduction 的完整闭环，并记录 static/EMA 的训练与资源开销。
