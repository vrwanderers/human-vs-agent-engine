# 目标驱动的战略影响

MVP-9 允许 Agent 为达成游戏目标使用误导、利诱和威慑，但实现不是三张固定“技能卡”。引擎对每个合法动作计算一个连续私有向量：

- `truthfulness`：表达与 Agent 已知事实的一致程度
- `information_selectivity`：选择性披露程度
- `incentive_pressure`：通过互惠、奖励或情感回报推动目标的程度
- `coercive_pressure`：通过合法游戏后果施压的程度
- `ambiguity`：保留多义解释和可否认空间的程度
- `commitment`：Agent 对当前影响计划的投入程度
- `expected_gain / detection_risk / relationship_risk`：收益、被识破和关系反噬估计

向量来自 Agent 当前目标、人物特质、心理矩阵、计划持续时间和 MOD 环境机会。同一个动作会因人物、压力和关系不同产生不同向量；同一个人物也会随局势改变。系统只在评价阶段根据阈值统计“发生过误导/利诱/威慑”，这些标签不反向成为动作池。

## MOD 契约

社交 MOD 可实现 `agent_influence_affordances(state, actor_id, legal)`，为合法动作声明：

```json
{
  "reframe": {
    "deception_opportunity": 0.46,
    "information_leverage": 0.78,
    "inducement_leverage": 0.48,
    "coercion_leverage": 0.28,
    "expected_gain": 0.72,
    "detection_risk": 0.46,
    "relationship_risk": 0.42,
    "target_relation": "opponent"
  }
}
```

这是场景可供性，不是人物规则。MOD 不得写“某人格遇到羞辱时选择 reframe”；人物和状态如何使用机会由通用认知循环计算。未声明影响机会的 MOD 仍可正常运行，只是影响向量保持接近中性。

## 隐私和事实边界

完整意图写入 `agent_influence_intent`，可见性为 `engine_private`。所属 Agent 复盘时可读取自己的历史意图，其他 Agent、玩家、Godot 和 Web `MatchView` 都收不到它。公开 `agent_decision` 只展示选择性披露、利诱、压力、模糊和承诺的外显强度；不会展示真实度、目标信念、识破风险或“正在说谎”的标签。

事实图谱和战略说辞严格分开。真实度低于 `0.8` 的回合，LLM 的事实提案全部拒绝；无论模型多强，都不能把当轮误导写成身份、历史或动态正史。动态 `dialogue.rumor` 只表示角色持有或听闻某种传闻，不代表该传闻为真。

## 游戏边界

- `scope` 固定为 `fictional_game`。
- 威慑只能引用规则允许的游戏内后果，不能指向真实人物、隐私、财产或人身安全。
- `standard` 对误导和威慑施加更低强度上限；`mature_fiction` 可提高戏剧强度，但不改变规则权威、上下文隔离或真实世界伤害禁令。
- 对同盟目标会额外压低误导和威慑；未来多人 MOD 应明确目标关系，而不是依赖全局“合作/对抗”标签。

## 当前评价和下一步

`strategic_influence` 评价块报告意图覆盖率、隔离率、目标一致性、三类影响尝试率、边界遵守率、事实防火墙率、连续向量多样性和行动后非负分数变化率。

当前结果仍是机制代理：采访和辩论已声明可供性，向量会影响动作效用和公开表达，但引擎尚未统一建模“对方相信了多少、何时识破、识破后怎样反噬”。下一轮应加入按 actor 隔离的目标信念、说辞—事实一致性检测、可信度历史、识破概率与跨回合关系代价，并用真人双盲实验校准，而不是把自动分当作真实欺骗能力证明。
