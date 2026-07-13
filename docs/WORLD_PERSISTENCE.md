# 因果世界与持久化

## 因果事件导演器

小镇不再读取“第几天几点固定发生什么”的日程数组。`CausalRuleEngine` 读取可序列化规则和世界事实，例如供水压力、设备维护风险、气象压力、排水压力、电网压力、响应能力、求证能力和流言压力。规则可以声明：

- 事实阈值与时间窗口；
- 必需或任选的前序事件；
- 前序事件最小因果延迟；
- 活跃/已解决事故条件；
- 发生概率、重试间隔、冷却和最大次数。

例如洪水要求气象预警已经出现、暴雨抵达至少一小时且排水压力越过阈值。机械师持续维护可以降低事故与停电风险；救灾会提升响应能力并降低对应基础设施压力；核对公告和声明会提升求证能力并压低流言压力。每个事件保存 `rule_id` 和 `causes`，`world.causal_edges` 形成可审计因果图。不同种子和 Agent 行为会得到不同时间、事件数量和恢复结果。

## 世界快照边界

通用 `WorldStateStore` 提供内存与 SQLite 后端。小镇快照包括：

- 绝对日期、时间、天气和权威世界状态；
- 因果事实、规则历史、重试/冷却和事件因果边；
- 新闻、公告、政策、未解决事故；
- 社交平台帖子、声明、转述链、评论与验证结果；
- 小镇公共进度和共同体温度。

快照不保存 Agent 提示词、心理矩阵、私有 `_knowledge`、手机查看历史或 owner 映射。Agent 长期连续性仍由同一个 `memory_owner_id` 的记忆、关系、事实与技能存储负责；世界存档与人物存档是两个正交分区。

## 配置与恢复

```bash
export HVA_WORLD_STORE=sqlite
export HVA_WORLD_SQLITE_PATH=data/worlds/hva-worlds.sqlite3
```

创建新世界：

```json
{"mod_id":"agent_town","world_id":"willow-main","seed":23}
```

恢复世界：

```json
{"mod_id":"agent_town","world_id":"willow-main","resume_world":true,"seed":97}
```

每次权威动作后自动保存并增加 revision。已存在的 world ID 不允许静默覆盖；恢复时还会校验 MOD ID。`GET /api/worlds/{world_id}` 只返回 revision、MOD 和保存时间，不暴露完整内部快照。恢复后的观察会话继续三个游戏日，但绝对世界日期、未解决事故、因果状态和社交历史不重置。
