# LLM 兼容层与上下文策略

## Provider 接口

`LLMProvider` 只定义一个异步 `complete(LLMRequest)` 契约。内置的 `OpenAICompatibleProvider` 对接 `/chat/completions`，适用于 OpenAI-compatible 服务，例如 DeepSeek、Groq、本地 Ollama 网关或企业代理。

环境变量示例见 `.env.example`：

```python
from hva_engine.llm import OpenAICompatibleProvider, ProviderRegistry

provider = OpenAICompatibleProvider.from_env()
registry = ProviderRegistry()
registry.register(provider)
```

Provider 返回的文本不会直接成为游戏动作。`LLMDecisionClient` 接受动作索引、简短可观察理由、可公开角色台词和最多五条事实提案，并再次检查动作索引。事实提案仍要经过 `AgentFactGraph` 的谓词白名单、依据、冲突和修订校验。采访台词可以进入公开 transcript，但不能改变动作类型或规则数值。LLM、启发式策略、回放策略都不能越过引擎规则层。

不支持 `response_format` 的本地或 unrestricted fictional-style Provider，可设置 `HVA_LLM_SUPPORTS_RESPONSE_FORMAT=false`。这只改变 Provider 请求格式，不会降低动作、事实、隐私或真实世界伤害边界。

调试后端提供同步 `chat/completions` 调用路径，便于现有同步引擎运行真实模型；生产部署仍应把远程推理迁移到异步任务/隔离进程，避免阻塞 FastAPI worker。

## 真实模型运行

设置：

- `HVA_AGENT_RUNTIME=llm`
- `HVA_LLM_BASE_URL / HVA_LLM_MODEL / HVA_LLM_API_KEY`
- `HVA_LLM_MODS=adversarial_interview`，限制哪些 MOD 使用远程模型
- `HVA_LLM_TEMPERATURE / HVA_LLM_MAX_TOKENS`
- `HVA_LLM_FALLBACK=false`，严格实验中禁止失败后使用基线 Agent

运行 `hva-llm-smoke` 会完成六轮采访。每次决策事件记录 Provider、模型、usage、事实提案接受/拒绝结果和公开台词；不会保存原始 Provider 响应、私有思维链或密钥。

结构化输出协议：

```json
{
  "action_index": 0,
  "reason": "brief observable summary",
  "utterance": "public in-character answer",
  "response_plan": {
    "strategy_weights": {
      "answer_honestly": 0.4,
      "set_boundary": 0.25,
      "counterattack": 0.2,
      "deflect_with_humor": 0.15
    },
    "intensity": 0.72,
    "emotional_display": "controlled_anger",
    "stance_tags": ["direct", "wounded"],
    "reveal_fact_ids": ["fact-0008"]
  },
  "fact_proposals": [
    {
      "subject": "self",
      "predicate": "belief.interpretation",
      "object": {"summary": "..."},
      "basis_fact_ids": ["fact-0001"]
    }
  ]
}
```

`action_index` 仍是规则校验用的主要策略；`strategy_weights` 只能引用当前合法动作，最多保留四种并归一化。揭露请求也不是直接写权限，只有满足剧情节奏且指向既有形成性记忆时才会产生公开 `story_reveal`。

## 提示词分层

每次决策构建独立 `ContextPacket`，层级固定为：

1. `runtime_contract`：运行边界、观察值不可信、禁止泄露私有上下文
2. `game_rules`：MOD 与引擎权威、只能选择合法动作
3. `model_boundary`：模型能力与引擎策略分离
4. `agent_role`：对抗/协作角色和对局身份
5. `fictional_identity` / `stable_persona`：私有自传、价值观和稳定人格
6. `shared_facts` / `compressed_private_memory`：团队事实与按显著性检索的私有记忆
7. `canonical_fact_graph`：当前可用事实及自由发挥约束
8. `appraisal_and_coping` / `situation_activated_traits`：评价、应对、心理矩阵和情境人格
9. `semantic_reflections` / `persistent_plan`：引用情景证据的反思和跨回合计划
10. `social_beliefs`：可错的信任、尊重、敌意、真诚度和对手行为信念
11. `narrative_dynamics`：竞争动机、承诺、秘密压力、身份失调、冲动压力、社会易感性、自我许可和行动后果遗留
12. `current_observation` / `deliberation_protocol`：观察、快慢模式与有限理性协议
13. `legal_actions`：规范化动作列表及 JSON 输出协议

系统/规则/角色进入 system message，其余进入 user message。观察值使用“不可信数据”标记，防止游戏文本或直播输入覆盖上层指令。

## 隔离与共享

- 每个 `AgentBrain` 持有独立的四类记忆系统与独立 `ContextPacket`。
- 对手看不到彼此的私有记忆、提示词或决策理由。
- 协作 Agent 只能通过 `SharedBlackboard` 共享事实。
- 黑板记录“谁执行了什么、出现了哪些规则事件”，不记录 chain-of-thought。
- 完整自传与未揭露事实只属于当前 Agent；公开故事必须通过 `story_reveal` 事件。
- 决策事件只保存摘要和可验证特征，明确不请求、不保存私密思维链。
- `context-preview` 是开发调试端点；公开部署必须在网关层关闭或加入管理员鉴权。

## 上下文压缩

默认上下文预算为 12,000 字符，私有检索记忆预算为 2,400 字符。超出时：

- 固定保留系统安全、规则、角色、当前观察和合法动作；
- 认知层先按时近性、重要性、相关性和情绪一致性选择至多 4 条情景记忆；
- 只把检索结果注入 Provider，未选中的私有经历不会因为靠近上下文尾部而自动进入；
- 事实图谱、评价/应对、反思、计划、社会信念、人物动力、当前观察和合法动作分别分配字符预算；
- 总预算触顶时逐层压缩内容，但保留所有层级标题和首尾证据，不再从整段尾部截断；
- 不用另一个 LLM 做摘要，避免摘要成本、漂移和跨 Agent 泄露；
- 诊断字段记录是否压缩、压缩前条数、共享事实数和最终字符量。

后续可增加可插拔 `ContextCompressor`，但压缩器输入必须限定为单个 Agent 的私有分区；团队摘要则只能消费共享黑板。

## 接入非兼容 Provider

实现 `LLMProvider.complete` 并注册即可。适配器负责把统一 messages、温度、token 限额和 JSON 格式要求翻译成厂商协议；鉴权、重试、限流和熔断留在 Provider 层，游戏引擎只消费标准化响应与 usage。
