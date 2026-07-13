# 人物卡：身份先验，不是决策脚本

## 两类数据严格分离

| 数据 | 用途 | 是否进入实时决策 | 是否包含动作表 |
|---|---|---:|---:|
| 叙事校准案例 | 离线检验通用认知机制能否解释人物关键抉择 | 否 | 候选项只属于评测题，不可复用为运行时动作 |
| 人物卡 | 为一名 Agent 提供稳定身份、经历、人格、动机与承诺先验 | 是 | 否；schema 明确禁止额外字段 |

文学资料的首要作用是暴露通用机制缺口。例如角色可能因为羞耻、依恋、身份威胁、诱惑或社会压力而做出并非静态人格最优的选择。引擎应把这些现象抽象为可跨作品、跨 MOD 复用的状态更新和效用因子，而不是把作品中的选择编译成 `if 条件 → 动作`。

## 运行时决策流

```mermaid
flowchart LR
  A[内置或玩家人物卡] --> B[稳定身份 / 人格 / 动机 / 承诺 / 形成性与生活回忆]
  B --> C[当前世界状态与事实图谱]
  C --> D[评价、情绪、社会信念与记忆检索]
  D --> E[动机冲突、后果债务与有限理性效用]
  E --> F[仅在 MOD 当前合法动作中选择]
  F --> G[结果进入记忆并更新人物弧光]
```

人物卡没有优先动作、问题—回答映射、固定台词、提示词或决策树。相同人物在不同世界状态、历史记忆、关系、压力和随机种子下可以作出不同选择；不同人物面对相同输入也会因动机、应对方式和承诺不同形成可辨识轨迹。

人物卡字段包括：

- 背景、愿望、核心创伤、价值观和社交风格；
- Big Five 与风险、损失厌恶、耐心、同理心、适应性、决策噪声等稳定倾向；
- 三段原创形成性记忆及其情绪效价和学习结果；
- 最多二十段可选生活回忆，可标注人物、家庭关系、主题、地点和时期；
- 结构化语言风格：话语体、教育口吻、词汇/句式复杂度、直接程度、粗粝度、温度、幽默、哲学抽象、技术术语和篇幅；
- 通用动机权重：自保、真相、归属、自主、地位、责任、救赎、关怀；
- 对价值、关系、隐私、自尊等抽象承诺的权重；
- 来源、原作语言、文化区域和版权策略元数据。

## 内置与玩家输入

`GET /api/character-cards` 返回可选内置人物卡。目前包括窦娥、孙悟空、鸳鸯、阿Q、Segismundo；所有内容均为原创抽象转述，不含原作文本。选择人物卡：

```json
{
  "mod_id": "adversarial_interview",
  "seed": 19,
  "agent_characters": [{"card_id": "ah_q"}]
}
```

玩家也可在 `agent_characters[].custom_card` 中提交完整 `CharacterCardSpec`。自定义卡只在本局生效，不会写入全局目录。Agent 对战或协作时，数组按 Agent 席位顺序分配，最多两张。

生活回忆和语言风格示例：

```json
{
  "lived_memories": [{
    "title": "雨后公园的一天",
    "recollection": "我和伴侣带女儿去河边公园，在旧树下分吃饭团。",
    "emotional_valence": 0.88,
    "lesson": "平凡陪伴比漂亮承诺可靠。",
    "people": ["伴侣", "女儿阿禾"],
    "themes": ["家庭", "子女", "陪伴"],
    "place": "河边公园",
    "time_period": "女儿七岁时的春天"
  }],
  "speech_style": {
    "voice_register": "plain",
    "education_voice": "limited_formal_schooling_but_experienced",
    "vocabulary_complexity": 0.22,
    "sentence_complexity": 0.25,
    "directness": 0.82,
    "roughness": 0.7,
    "warmth": 0.68,
    "humor": 0.18,
    "philosophical_abstraction": 0.08,
    "technical_jargon": 0.02,
    "verbosity": 0.28,
    "verbal_habits": ["用生活经验解释抽象问题"]
  }
}
```

`voice_register` 支持 `neutral/plain/colloquial/formal/academic/philosophical/technical/aristocratic/confrontational`。这些字段只控制措辞，不指定立场、策略或动作。真实 Provider 在身份层收到约束；基线表达器对 MOD 提供的语义文本应用相同表面风格。粗粝度在 `standard` 模式受限，只有 `mature_fiction` 可以使用更粗俗的虚构措辞。

## 提示词与规则安全

人物卡按不可信声明式数据处理：

- Pydantic 使用 `extra=forbid`；`action_rules`、固定响应池或未知字段会直接得到 422；
- 人物卡进入 L5 身份层，不能覆盖 L1–L4 的运行契约、游戏规则、模型边界和 Agent 角色；
- 卡片不直接写世界状态、事实图谱或动作；
- 生活回忆作为用户声明的虚构正史进入不可变私有事实和长期索引；模型不能自行增加家庭成员；
- 最终动作仍必须完整匹配 MOD 的 `legal_actions`；
- 自定义卡只属于对应 Agent 的私有上下文，协作共享仍只经过净化事实黑板。

## 中文近现代材料与版权门禁

鲁迅《阿Q正传》等公版材料可形成可分发的人物卡。茅盾、路遥、莫言等版权仍有效或状态复杂的作品，仓库默认只保存短篇原创机制校准标注和来源元数据，不分发详细人物卡、原文、台词、模仿性文风或情节复刻。取得授权后可通过本地 `licensed_annotation` 数据或玩家自定义人物卡接入，不应把授权内容提交到公共仓库。

MVP-8 的事件记录会保存人物卡 ID、身份层 ID 和 `runtime_cognition_not_scripted_actions` 决策模型标识。评价器报告 `character_card_grounding.identity_grounding_rate`，但这只是身份贯通检查；角色忠实度、真人感和表演质量仍需独立人类盲评。
