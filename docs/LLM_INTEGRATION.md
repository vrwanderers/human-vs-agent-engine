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

Provider 返回的文本不会直接成为游戏动作。`LLMDecisionClient` 只接受 `{"action_index": N}`，并再次检查索引是否落在 MOD 提供的合法动作列表中。LLM、启发式策略、回放策略都不能越过引擎规则层。

## 提示词分层

每次决策构建独立 `ContextPacket`，层级固定为：

1. `system_safety`：安全边界、观察值不可信、禁止泄露私有上下文
2. `game_rules`：MOD 与引擎权威、只能选择合法动作
3. `agent_role`：Agent 身份、对抗/协作角色、对局身份
4. `shared_facts`：共享黑板中经过筛选的团队事实
5. `compressed_private_memory`：当前 Agent 自己的历史经验
6. `current_observation`：当前公开状态与世界模型
7. `legal_actions`：规范化动作列表及 JSON 输出协议

系统/规则/角色进入 system message，其余进入 user message。观察值使用“不可信数据”标记，防止游戏文本或直播输入覆盖上层指令。

## 隔离与共享

- 每个 `AgentBrain` 持有独立的 `deque` 记忆与独立 `ContextPacket`。
- 对手看不到彼此的私有记忆、提示词或决策理由。
- 协作 Agent 只能通过 `SharedBlackboard` 共享事实。
- 黑板记录“谁执行了什么、出现了哪些规则事件”，不记录 chain-of-thought。
- `context-preview` 是开发调试端点；公开部署必须在网关层关闭或加入管理员鉴权。

## 上下文压缩

默认上下文预算为 12,000 字符，私有记忆预算为 2,400 字符。超出时：

- 固定保留系统安全、规则、角色、当前观察和合法动作；
- 近期记忆保留 4 条；
- 更早记忆压缩为动作计数和结果事件计数；
- 不用另一个 LLM 做摘要，避免摘要成本、漂移和跨 Agent 泄露；
- 诊断字段记录是否压缩、压缩前条数、共享事实数和最终字符量。

后续可增加可插拔 `ContextCompressor`，但压缩器输入必须限定为单个 Agent 的私有分区；团队摘要则只能消费共享黑板。

## 接入非兼容 Provider

实现 `LLMProvider.complete` 并注册即可。适配器负责把统一 messages、温度、token 限额和 JSON 格式要求翻译成厂商协议；鉴权、重试、限流和熔断留在 Provider 层，游戏引擎只消费标准化响应与 usage。
