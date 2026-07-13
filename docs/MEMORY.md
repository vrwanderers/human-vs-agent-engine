# 长期与短期记忆

## 生命周期

记忆不是完整对话文件。每次行动结果先进入短期层，短期层同时受两个边界控制：

- `short_term_ttl_turns`：默认 6 回合；超过期限且未提升的经历会遗忘。
- `short_term_limit`：默认 16 条；高频事件不会无限占用运行内存。

系统根据分数变化、惊讶和情绪强度计算重要性。达到阈值，或涉及身份、家庭、关系、强惊讶时，经历被提升到长期层。提升时生成不超过 180 字符的摘要，并分类为 `experience`、`strategy`、`identity`、`family`、`relationship`、`emotion`、`world` 等一个或多个维度。中文使用字符双元词项，因此“童年身份”“女儿公园”等内容无需外部分词服务也可以索引。

## 自传与生活回忆

人物卡的三段 `formative_memories` 仍负责主要人物弧光。新增 `lived_memories` 用于更细的私生活和日常经历，例如家庭成员、子女、伴侣、一次出游、地点、时期和从中形成的感受。每条生活回忆包含标题、第一人称回忆、情绪效价和经验教训，还可提供 `people`、`themes`、`place`、`time_period` 索引字段。

Agent 创建时，背景、形成性记忆和生活回忆同时写入私有长期索引，并在事实图谱中建立不可变 `history.lived_memory.*` 节点。完整生活回忆不会每轮全部塞入提示词；当前问题中的主题、问题文本和动作会形成检索 cue，只注入最相关项。这样 Agent 可以在谈到家庭或子女时回忆具体经历，又不能现场补写不存在的孩子或往事。

## 关系、印象和他人模型

每个记忆 owner 对每个玩家/Agent 单独维护 `RelationshipProfile`，包括信任、尊重、温暖、敌意、熟悉度、当前态度、行为计数、主观印象、公开披露的背景和敏感主题。它不是单一好感度：

- `observed`：亲眼看到的动作或问题，保存来源 event 和 memory ID；
- `inferred_pattern`：从重复行为形成、可修订的印象；
- `publicly_reported`：对方通过公开故事事件披露的背景，只表示“对方这样说过”；
- `sensitive_points`：分为对方反复关注的主题和观察到的反应触发点，明确不是已证实弱点。

关系档案使用 target 哈希标签做索引，并保存为 `semantic/relationship_profile`；每次互动证据则保存为独立情景记忆。关系摘要也写入 `belief.relationship_impression`，谓词本身明确它是信念而非对方的正史事实。

## 索引检索

长期存储实现 `LongTermMemoryStore`，以 `owner_id` 为首要分区。检索分两步：

1. 通过词项、分类、标签倒排索引取命中项，并加入一个有上限的最近记忆窗口。
2. 只对候选集合计算相关性、时近性、重要性和情绪一致性，最多把 4 条结果注入 Agent 上下文。

管理员诊断中的 `last_query` 会报告总记录数、索引命中数、加载候选数和 `full_scan=false`，用于确认增长到长期历史后没有退化成全表遍历。普通程序经验通过长期记录聚合为动作经验值；技能自动化另存为 owner-scoped 的 `procedural_skill` 文档，按技能族保存各场景的执行次数、成功/失败、惊讶、指导执行与自动执行次数。它不会因一次成功直接接管，也不会与其他 Agent 共享。证据反思存为 `semantic` 记忆，并保留引用的情景 memory ID。

## 后端

默认配置适合测试和单进程调试：

```env
HVA_MEMORY_STORE=memory
```

需要跨进程、跨对局恢复时使用 SQLite：

```env
HVA_MEMORY_STORE=sqlite
HVA_MEMORY_SQLITE_PATH=data/memory/hva-memory.sqlite3
```

SQLite 不是把历史写进一个 JSON 字段。它使用 `memories`、`memory_terms`、`memory_categories`、`memory_tags` 和 `memory_sequences` 规范化表，并为 owner/turn、owner/action 及三个倒排维度建立数据库索引。`LongTermMemoryStore` 是可替换边界，后续可以接入 PostgreSQL、向量检索或分布式存储，而不改变 Agent 认知循环。

## 跨对局身份与隔离

人物卡定义“是谁”，`memory_owner_id` 定义“哪一份私有经历”。调用方只有显式提供同一稳定 ID，后续对局才会恢复该角色经历：

```json
{
  "card_id": "dou_e",
  "memory_owner_id": "dou-e-campaign-01"
}
```

未提供时，引擎使用随机生成的对局内 Agent ID，因此默认不会跨局串记忆。一个对局内不允许两个 Agent 使用相同 `memory_owner_id`。所有查询、写入、反思、程序经验和技能熟练度都必须携带 owner；协作共享仍只能走 `SharedBlackboard`，不会把私有长期记忆或熟练技能复制给队友。

如需让 Agent 在后续对局认出同一位真人，还要在创建对局时提供稳定的 `human_memory_id`，例如 `{"human_name":"Alice","human_memory_id":"human-alice"}`。该 ID 只用于私有关系分区，不进入公开事件。

## 当前边界

- 当前摘要和分类器是可复现的规则基线，避免为了存一条记忆再调用 LLM；后续可增加受 schema 和事实图谱约束的摘要器。
- 重要性提升目前是确定性阈值，尚未实现睡眠式离线巩固、合并近重复记忆和时间跨度压缩。
- 他人背景目前只从引擎验证过的公开故事事件提取；自由文本中的自我陈述仍需增加“声明—佐证—冲突”验证流程。
- SQLite 适合单机原型。高并发服务需要实现同一接口的服务型数据库后端，并增加租户鉴权、加密、保留期和删除审计。
