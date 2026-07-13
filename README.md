# Human vs Agent Engine / 人类 VS Agent 游戏引擎

一个**评价体系驱动**的通用策略游戏引擎实验场：同一套事件流、Agent 接口与评估指标，可以承载战术对战、赛车策略、辩论，以及后续的宫廷政变、国际局势、病毒传播等 MOD。

当前版本是可运行的 MVP，包含：

- 通用回合制状态机、MOD 注册表和可替换 Agent 策略
- 6 维在线评价：玩家参与度、引擎通用性、动态性、虚拟玩家评价、AI 对手智能性、AI 人类感
- 5 个 MVP MOD：`tactical_duel`、`racing_strategy`、`debate_arena`、`crisis_coop`、`adversarial_interview`
- 论文驱动的 Agent 认知循环：四类记忆、证据反思、情绪评价/再评价、情境人格激活、社会信念、持续计划、快慢有限理性与结果复盘
- 慢变量人物动力学：竞争动机、关系承诺、秘密压力、身份失调、诱惑、社会施压、自我合理化与行动后果会跨回合改变偏好并推动人物弧光
- 连续困境与延迟代价：MOD 可为合法动作声明价值、关系、承诺、即时收益与延迟风险；选择会形成可修复但不会自动清零的债务，后续回合再结算后果
- 独立人物参考校准：用 38 个文学、戏剧、电影/剧本元数据和机构传记关键抉择案例检验通用机制；其中 11 个中文文学案例、4 个中文近现代小说案例，均不进入运行时动作池
- 声明式人物卡：可选择 5 张内置人物卡或提交玩家原创卡；卡片只提供身份、经历、人格、动机和承诺，不含固定动作、台词或问题—回答映射
- 渐进式角色故事揭露：关键回合、压力、受挫、信任和终局会产生 `story_reveal` 事件
- 连续混合应对：采访中的七类策略是内部行为坐标，每次回答混合 2–4 类策略及情绪强度，而不是传统 NPC 的固定单选招式
- 目标驱动的连续战略影响：Agent 可按目标、人格、压力、关系与场景机会组合选择性披露、误导、利诱和游戏内威慑；这些不是固定招式池
- 战略意图隔离：真实度、目标信念和识破风险仅写入所属 Agent 的引擎私有事件，玩家和其他 Agent 只能观察公开言行
- 受约束的事实图谱：核心身世不可覆盖，自由发挥必须引用事实依据，可变事实保留修订链
- 可选 Neo4j 持久化，存储 Agent、事实、揭露状态和 `SUPERSEDES` 关系
- Agent 私有上下文隔离、分层提示、确定性记忆压缩与协作共享黑板
- OpenAI-compatible LLM Provider 接口，可快速切换云端或本地模型
- 人类 vs Agent、Agent vs Agent、Agent 协作、人类-Agent 协作四种模式
- FastAPI 控制后端与浏览器调试面板
- Godot 4 展示客户端
- 直播弹幕命令入口与可扩展平台适配协议
- 自动化测试与 Docker 开发环境

## 快速开始

需要 Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn hva_engine.api:app --reload
```

打开 <http://127.0.0.1:8000> 使用调试控制台，或用 Godot 4 打开 `godot/project.godot`。

也可以使用 Docker：

```bash
docker compose up --build
```

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"tactical_duel","human_name":"Player","seed":7}'
```

Agent 对战和协作实验：

```bash
curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"debate_arena","mode":"agent_vs_agent","seed":7}'

curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"crisis_coop","mode":"agent_coop","seed":7}'
```

调整人类感和虚构角色的阴暗倾向（不改变引擎规则权威）：

```bash
curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"debate_arena","mode":"agent_vs_agent","seed":7,"agent_tuning":{"realism":0.85,"shadow_intensity":0.7,"content_mode":"mature_fiction"}}'
```

启动“逆风采访”，用尖锐问题测试 Agent 的心理与人物弧光：

```bash
curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"adversarial_interview","human_name":"Interviewer","seed":7}'
```

选择内置人物卡（先用 `GET /api/character-cards` 查看目录）：

```bash
curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"adversarial_interview","seed":19,"agent_characters":[{"card_id":"ah_q"}]}'
```

人物卡只作为稳定身份先验，实时动作仍由世界状态、私有记忆、心理矩阵、社会信念、人物动力和 MOD 合法动作共同推演。自定义人物卡使用同一 `CharacterCardSpec`，未知字段和 `action_rules` 会被拒绝。

Agent-only 对局会自动运行到终局。响应中的 `agent_summaries` 暴露世界模型、心理矩阵、公开身份片段、叙事进度和公开事实图谱；事件流中的 `agent_decision` 保存可观察的简短理由、置信度、预期效果和影响表现，但不保存私密思维链或战略底牌。引擎内部的 `agent_influence_intent` 事件不会进入 `MatchView`，且其他 Agent 只能收到公开事件和自己的私有意图。

默认 MVP 使用可复现的启发式 Agent，便于建立评价基线；`hva_engine.llm` 已提供分层上下文、Provider 注册表和严格动作索引解析，可在不改 MOD 规则的前提下替换为真实 LLM。

### 真实 LLM 采访测试

使用任意 OpenAI-compatible `/chat/completions` Provider。请通过环境或密钥管理器设置 API Key，不要写入仓库：

```bash
export HVA_AGENT_RUNTIME=llm
export HVA_LLM_PROVIDER=my-provider
export HVA_LLM_BASE_URL=https://your-provider.example/v1
export HVA_LLM_MODEL=your-model
export HVA_LLM_API_KEY=your-secret
export HVA_LLM_MODS=adversarial_interview

hva-llm-smoke --seed 7 --question-policy most_severe --character-card ah_q
```

未安装项目脚本时使用 `python -m hva_engine.llm_smoke`。默认不允许 smoke test 静默回退：六次回答必须全部来自真实 LLM，否则命令失败。输出包含公开问答、动作、token usage、心理轨迹、事实提案结果、人物弧光和完整评分，但不包含 API Key、私有提示词或思维链。

随后使用响应中的 `human_player_id` 提交动作：

```bash
curl -X POST http://127.0.0.1:8000/api/matches/MATCH_ID/actions \
  -H 'content-type: application/json' \
  -d '{"actor_id":"PLAYER_ID","action":{"type":"move","payload":{"direction":"right"}}}'
```

单局评价位于 `GET /api/matches/{id}/evaluation`；跨对局实验汇总位于 `GET /api/evaluations/summary`，会按 `MOD:模式` 分组比较综合分与六维分数。

## Neo4j 事实图谱

默认使用进程内存储。启用 Neo4j：

```bash
pip install -e '.[neo4j]'
export HVA_FACT_STORE=neo4j
export HVA_NEO4J_URI=neo4j://127.0.0.1:7687
export HVA_NEO4J_USER=neo4j
export HVA_NEO4J_PASSWORD=your-password
uvicorn hva_engine.api:app --reload
```

Docker 开发环境可用 `HVA_FACT_STORE=neo4j docker compose --profile neo4j up --build`。详细约束见 [事实图谱](docs/FACT_GRAPH.md)。

运行评分 MVP-9 的多种子镜像基准（对抗模式会交换双方席位）：

```bash
python -m hva_engine.benchmark --seeds 25
# 安装后也可使用：hva-benchmark --seeds 25
```

运行不含作品原文的叙事人物决策校准：

```bash
python -m hva_engine.narrative_calibration
# 安装后也可使用：hva-narrative-calibration
```

## 设计原则

1. **先评价，再扩展**：所有对局从第一回合开始产生同构事件和评分。
2. **MOD 是纯状态机**：规则与展示、传输、Agent 实现解耦。
3. **Agent 只能看到观察值**：为隐藏信息、远程模型与锦标赛留下边界。
4. **直播输入也是动作源**：弹幕与 Godot、Web 控制台共享同一校验链路。
5. **用 MVP 数据升级架构**：评价低分对应明确的下一轮改造方向。

详细内容见 [战略影响机制](docs/STRATEGIC_INFLUENCE.md)、[人物卡](docs/CHARACTER_CARDS.md)、[研究驱动的人类感 Agent](docs/RESEARCH_HUMAN_LIKE_AGENTS.md)、[叙事人物决策校准](docs/NARRATIVE_CHARACTER_CALIBRATION.md)、[评价体系](docs/EVALUATION.md)、[架构说明](docs/ARCHITECTURE.md)、[逆风采访 MOD](docs/INTERVIEW_MOD.md)、[事实图谱](docs/FACT_GRAPH.md) 与 [LLM/上下文接入](docs/LLM_INTEGRATION.md)。

## 弹幕命令

当前通用入口为 `POST /api/live/danmaku`，接受：

```json
{"match_id":"...", "user":"viewer42", "message":"!move right"}
```

支持的 MVP 命令包括 `!move up`、`!attack`、`!accelerate`、`!conserve`、`!pit`、`!evidence`、`!emotion`、`!rebuttal`，以及采访 MOD 的 `!ask identity`、`!ask failure` 等主题选择。生产环境中可在 `DanmakuAdapter` 前增加 Bilibili、抖音、Twitch 或 YouTube 的鉴权/签名适配器，并保留限流、投票聚合和内容安全层。

## 路线图

- M1：独立标注全新叙事人物 holdout，并采集真实玩家/Agent 双盲基线，分别校准叙事忠实度、真人感与欺骗识破/反噬曲线
- M2：加入并行回合、隐藏信息、回放与持久化
- M3：远程 LLM Agent 沙箱、Elo/Glicko 锦标赛和观众投票窗口
- M4：MOD SDK、资产协议、直播平台正式适配器与 Godot 可视化组件库

## License

Apache-2.0
