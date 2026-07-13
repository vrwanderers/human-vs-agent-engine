# Human vs Agent Engine / 人类 VS Agent 游戏引擎

一个**评价体系驱动**的通用策略游戏引擎实验场：同一套事件流、Agent 接口与评估指标，可以承载战术对战、赛车策略、辩论，以及后续的宫廷政变、国际局势、病毒传播等 MOD。

当前版本是可运行的 MVP，包含：

- 通用回合制状态机、MOD 注册表和可替换 Agent 策略
- 6 维在线评价：玩家参与度、引擎通用性、动态性、虚拟玩家评价、AI 对手智能性、AI 人类感
- 6 个 MVP MOD：`agent_town`、`tactical_duel`、`racing_strategy`、`debate_arena`、`crisis_coop`、`adversarial_interview`
- 论文驱动的 Agent 认知循环：TTL 短期记忆、摘要/分类/倒排索引长期记忆、证据反思、情绪评价/再评价、情境人格激活、社会信念、持续计划、快慢有限理性与结果复盘
- 连续决策倾向：每个合法动作得到可审计的吸引力、排名和主要动因；它只是动机压力，不是写死的候选答案，角色可在记忆、承诺或关系风险支持下作出有限理性的非最优选择
- 行动后心理再评价：规则结果会立即回写压力、恐惧、愤怒、士气与不确定性，形成“受压—应对—部分恢复—再次受压”的轨迹
- 慢变量人物动力学：竞争动机、关系承诺、秘密压力、身份失调、诱惑、社会施压、自我合理化与行动后果会跨回合改变偏好并推动人物弧光
- 连续困境与延迟代价：MOD 可为合法动作声明价值、关系、承诺、即时收益与延迟风险；选择会形成可修复但不会自动清零的债务，后续回合再结算后果
- 独立人物参考校准：用 38 个文学、戏剧、电影/剧本元数据和机构传记关键抉择案例检验通用机制；其中 11 个中文文学案例、4 个中文近现代小说案例，均不进入运行时动作池
- 声明式人物卡：可选择 5 张内置人物卡或提交玩家原创卡；卡片只提供身份、经历、人格、动机和承诺，不含固定动作、台词或问题—回答映射
- 可检索自传生活回忆：人物卡可注入家庭、子女、伴侣、地点和日常故事，作为私有正史写入事实图谱与长期索引，只有相关话题才召回
- 按对象的关系记忆：分别保存对每位玩家/Agent 的信任、态度、行为印象、公开背景和敏感主题，观察、推测与事实严格分层
- 人物语言风格层：朴实、口语、学术、哲学、技术、居高临下或粗粝等表达由连续参数约束，不改变动作选择，也不是固定台词池
- 渐进式角色故事揭露：关键回合、压力、受挫、信任和终局会产生 `story_reveal` 事件
- 连续混合应对：采访中的七类策略是内部行为坐标，每次回答混合 2–4 类策略及情绪强度，而不是传统 NPC 的固定单选招式
- 目标驱动的连续战略影响：Agent 可按目标、人格、压力、关系与场景机会组合选择性披露、误导、利诱和游戏内威慑；这些不是固定招式池
- 战略意图隔离：真实度、目标信念和识破风险仅写入所属 Agent 的引擎私有事件，玩家和其他 Agent 只能观察公开言行
- 受约束的事实图谱：核心身世不可覆盖，自由发挥必须引用事实依据，可变事实保留修订链
- 可选 Neo4j 持久化，存储 Agent、事实、揭露状态和 `SUPERSEDES` 关系
- 可选 SQLite 长期记忆库；记忆正文、摘要、分类、标签和词项使用规范化表及索引，不写成单个历史 JSON
- Agent 私有上下文隔离、分层提示、关键层优先的结构化压缩与协作共享黑板；被压缩的低优先级段仍保持合法 JSON 并可审计
- OpenAI-compatible LLM Provider 接口，可快速切换云端或本地模型
- 多模态刺激、身体反射、可成长/退化的场景化程序技能，以及必要时才调用 LLM 的审慎门控；陌生路线或新工作先规划，稳定练习后才允许本地自动接管
- 人类 vs Agent、Agent vs Agent、Agent 协作、人类-Agent 协作四种模式
- FastAPI 控制后端与浏览器调试面板
- Godot 4 展示客户端
- `Agent 小镇`权威世界模型：时间、天气、灾害、意外、新闻、公告、镇长讲话和政策按传播范围进入各 Agent 的独立观察；Agent 可救灾、避险、查公告和照顾邻居
- 因果事件导演器：世界风险、Agent 维护/救灾/求证行为、前序事件和冷却共同决定后续事件，不再使用固定回归日程
- MOD 通用社交媒体兼容层：微博式微帖、短视频、关系/信任排序、转述溯源、私有信念、主动调查与显式事实核查
- 可选 SQLite 世界快照：绝对时间、因果图、政策、事故和社交历史可跨对局恢复，不混入 Agent 私有上下文
- 基线 vs 真实 LLM 的无回退配对实验、A/B 条件盲化、真人 1–7 分采集和持久化汇总
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

Godot 默认打开 2D 俯视角 `Agent 小镇`：一名真人观察者与三名独立 AgentBrain 共同生活三天。每个镇民会从稳定 memory owner 生成私有职业背景、人格、自传和家庭回忆；世界导演持续发布天气、新闻、政策与突发事件。点击“等待 30 分钟”可以单步观察 Agent 的日常和危机反应，“自动观察”会连续推进。地图和角色使用 Godot 原生像素绘制，规则坐标与画面坐标一致。详见 [Agent 小镇](docs/AGENT_TOWN.md)。

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

跨对局延续同一角色的私有记忆时，为人物选择提供稳定的 `memory_owner_id`，并启用 SQLite。未提供该字段时使用对局内 Agent ID，不会意外串联不同对局：

```bash
export HVA_MEMORY_STORE=sqlite
export HVA_MEMORY_SQLITE_PATH=data/memory/hva-memory.sqlite3

curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"adversarial_interview","seed":19,"agent_characters":[{"card_id":"ah_q","memory_owner_id":"ah-q-campaign-01"}]}'
```

记忆的生命周期、索引和隔离契约见 [长期与短期记忆](docs/MEMORY.md)。

小镇可直接为三个 Agent 指定跨局稳定 owner，不必绑定文学人物卡。相同 owner 会重新生成同一套人物先验，并在启用 SQLite 后延续该人物的长期经历；不同 owner 的记忆、关系和熟练技能互不读取：

```bash
curl -X POST http://127.0.0.1:8000/api/matches \
  -H 'content-type: application/json' \
  -d '{"mod_id":"agent_town","seed":23,"agent_memory_owner_ids":["willow-astra","willow-nova","willow-mira"]}'
```

因果世界存档、社交媒体兼容层和真人盲评流程分别见 [因果世界与持久化](docs/WORLD_PERSISTENCE.md)、[社交媒体兼容层](docs/SOCIAL_MEDIA.md) 与 [真实 LLM 自然度盲评](docs/BLIND_NATURALNESS_EVAL.md)。

人物卡只作为稳定身份先验，实时动作仍由世界状态、私有记忆、心理矩阵、社会信念、人物动力和 MOD 合法动作共同推演。自定义人物卡使用同一 `CharacterCardSpec`，未知字段和 `action_rules` 会被拒绝。

Agent-only 对局会自动运行到终局。公开 `MatchView` 中的 `agent_summaries` 只包含已揭露身份、叙事进度和公开事实；心理矩阵、记忆、计划、对手模型、完整 `agent_decision` 与 `agent_influence_intent` 都属于引擎私有数据。公开动作还会移除 `response_plan` 等内部注解。可信开发环境可设置 `HVA_DEBUG_TOKEN`，再通过 `X-HVA-Debug-Token` 请求管理员调试视图或 `context-preview`。

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

需要逐步检查每次决策时，可使用严格的 stdin/stdout 调试桥：

```bash
hva-llm-step-debug --seed 19 --question-policy most_severe \
  --character-card ah_q --context-output full \
  --report /tmp/hva-step-report.json
```

每轮会先冻结并输出该 Agent 的私有快照、合法动作和可选完整分层提示，然后停下来等待一行 LLM JSON。格式错误会在当前步返回 `llm_decision_rejected` 并允许重试；回退策略始终关闭。此命令及生成的报告含私有调试数据，只能在可信开发环境使用。

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

运行评分 MVP-11 的多种子镜像基准（对抗模式会交换双方席位）：

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

详细内容见 [MVP-10 窦娥单步 LLM 测试](docs/MVP10_DOU_E_LLM_TEST.md)、[战略影响机制](docs/STRATEGIC_INFLUENCE.md)、[多模态刺激、反射与审慎门控](docs/IMPLICIT_CONTROL.md)、[人物卡](docs/CHARACTER_CARDS.md)、[研究驱动的人类感 Agent](docs/RESEARCH_HUMAN_LIKE_AGENTS.md)、[叙事人物决策校准](docs/NARRATIVE_CHARACTER_CALIBRATION.md)、[评价体系](docs/EVALUATION.md)、[架构说明](docs/ARCHITECTURE.md)、[逆风采访 MOD](docs/INTERVIEW_MOD.md)、[事实图谱](docs/FACT_GRAPH.md) 与 [LLM/上下文接入](docs/LLM_INTEGRATION.md)。

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
