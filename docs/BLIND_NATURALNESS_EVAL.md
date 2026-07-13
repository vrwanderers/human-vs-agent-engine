# 真实 LLM 与真人自然度盲评

引擎自动分只能证明结构出现，不能判断 Agent 是否“像真人”。自然度校准使用同种子、同人物 owner、同人类等待动作的配对实验：一条轨迹使用可复现基线，另一条使用真实 OpenAI-compatible Provider；随后隐藏条件、Provider、模型和引擎代理分，只向评价者展示公开行动、对话、帖子、求证和世界事件。

## 生成严格配对任务

```bash
export HVA_LLM_BASE_URL=https://provider.example/v1
export HVA_LLM_MODEL=model-name
export HVA_LLM_API_KEY=secret

hva-town-naturalness-calibration \
  --seed 23 --rounds 8 \
  --study-id agent-town-naturalness-v1 \
  --database data/evaluation/hva-blind.sqlite3
```

该命令固定 `llm_fallback=false`。缺少 Provider 配置、请求失败或返回非法动作时实验直接失败，不会用本地策略补齐后冒充真实 LLM。输出给出盲评 trial ID，但不显示 A/B 条件对应关系。

生产盲评服务使用同一 SQLite：

```bash
export HVA_BLIND_EVAL_STORE=sqlite
export HVA_BLIND_EVAL_SQLITE_PATH=data/evaluation/hva-blind.sqlite3
uvicorn hva_engine.api:app
```

评价者打开：

```text
http://127.0.0.1:8000/blind-eval?trial=trial-...
```

每个样本分别用 1–7 分评价反应自然度、人物一致性、情境贴合和戏剧趣味，并选择 A、B 或难以区分。评价者 ID 只保存 SHA-256 截断哈希；同一评价者不能重复评价同一 trial。

`GET /api/evaluations/blind-summary/{study_id}` 会按真实隐藏条件聚合均值和偏好胜场。少于 12 份评分时固定标记 `insufficient_human_ratings`。正式结论还应使用多个种子、随机化样本顺序、文化/语言分层评价者和置信区间；当前 12 份只是让指标不再处于完全空白状态的最低门槛。

本仓库测试中的假 Provider 只验证上下文隔离、严格无回退和盲化协议，不能记为真实模型结果。只有上述命令成功调用外部 Provider，并收到真实人工评分后，才能报告自然度校准结果。

建议每个种子至少收集 12 份独立评价，并跨多个种子复测。引擎代理分与盲评结果必须分栏报告；`player_experience` 在没有真人样本时保持 `null`，不能用规则覆盖率、语言长度或当前操作者的主观印象代填。
