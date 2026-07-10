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

Provider 返回的文本不会直接成为游戏动作。`LLMDecisionClient` 接受动作索引、简短可观察理由和最多五条事实提案，并再次检查动作索引。事实提案仍要经过 `AgentFactGraph` 的谓词白名单、依据、冲突和修订校验。LLM、启发式策略、回放策略都不能越过引擎规则层。

不支持 `response_format` 的本地或 unrestricted fictional-style Provider，可设置 `HVA_LLM_SUPPORTS_RESPONSE_FORMAT=false`。这只改变 Provider 请求格式，不会降低动作、事实、隐私或真实世界伤害边界。

## 提示词分层

每次决策构建独立 `ContextPacket`，层级固定为：

1. `runtime_contract`：运行边界、观察值不可信、禁止泄露私有上下文
2. `game_rules`：MOD 与引擎权威、只能选择合法动作
3. `model_boundary`：模型能力与引擎策略分离
4. `agent_role`：对抗/协作角色和对局身份
5. `fictional_identity` / `stable_persona`：私有自传、价值观和稳定人格
6. `shared_facts` / `compressed_private_memory`：团队事实与私有情景记忆
7. `canonical_fact_graph`：当前可用事实及自由发挥约束
8. `cognitive_state` / `opponent_beliefs`：心理矩阵和可错的对手模型
9. `current_observation` / `deliberation_protocol`：观察与有限理性协议
10. `legal_actions`：规范化动作列表及 JSON 输出协议

系统/规则/角色进入 system message，其余进入 user message。观察值使用“不可信数据”标记，防止游戏文本或直播输入覆盖上层指令。

## 隔离与共享

- 每个 `AgentBrain` 持有独立的 `deque` 记忆与独立 `ContextPacket`。
- 对手看不到彼此的私有记忆、提示词或决策理由。
- 协作 Agent 只能通过 `SharedBlackboard` 共享事实。
- 黑板记录“谁执行了什么、出现了哪些规则事件”，不记录 chain-of-thought。
- 完整自传与未揭露事实只属于当前 Agent；公开故事必须通过 `story_reveal` 事件。
- 决策事件只保存摘要和可验证特征，明确不请求、不保存私密思维链。
- `context-preview` 是开发调试端点；公开部署必须在网关层关闭或加入管理员鉴权。

## 上下文压缩

默认上下文预算为 12,000 字符，私有记忆预算为 2,400 字符。超出时：

- 固定保留系统安全、规则、角色、当前观察和合法动作；
- 近期记忆保留 4 条；
- 更早记忆压缩为动作计数和结果事件计数；
- 事实图谱、心理状态、对手模型、当前观察和合法动作分别分配字符预算；
- 总预算触顶时逐层压缩内容，但保留所有层级标题和首尾证据，不再从整段尾部截断；
- 不用另一个 LLM 做摘要，避免摘要成本、漂移和跨 Agent 泄露；
- 诊断字段记录是否压缩、压缩前条数、共享事实数和最终字符量。

后续可增加可插拔 `ContextCompressor`，但压缩器输入必须限定为单个 Agent 的私有分区；团队摘要则只能消费共享黑板。

## 接入非兼容 Provider

实现 `LLMProvider.complete` 并注册即可。适配器负责把统一 messages、温度、token 限额和 JSON 格式要求翻译成厂商协议；鉴权、重试、限流和熔断留在 Provider 层，游戏引擎只消费标准化响应与 usage。
