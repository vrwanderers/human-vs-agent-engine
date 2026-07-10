# Human vs Agent Engine / 人类 VS Agent 游戏引擎

一个**评价体系驱动**的通用策略游戏引擎实验场：同一套事件流、Agent 接口与评估指标，可以承载战术对战、赛车策略、辩论，以及后续的宫廷政变、国际局势、病毒传播等 MOD。

当前版本是可运行的 MVP，包含：

- 通用回合制状态机、MOD 注册表和可替换 Agent 策略
- 5 维在线评价：玩家参与度、引擎通用性、动态性、虚拟玩家评价、AI 对手智能性
- 4 个 MVP MOD：`tactical_duel`、`racing_strategy`、`debate_arena`、`crisis_coop`
- 显式 Agent 认知循环：观察、世界模型、有限记忆、决策解释、规则硬约束
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

Agent-only 对局会自动运行到终局。响应中的 `agent_summaries` 暴露每个 Agent 的当前世界模型、决策次数和近期记忆，事件流中的 `agent_decision` 保存决策理由、置信度与预期效果。

默认 MVP 使用可复现的启发式 Agent，便于建立评价基线；`hva_engine.llm` 已提供分层上下文、Provider 注册表和严格动作索引解析，可在不改 MOD 规则的前提下替换为真实 LLM。

随后使用响应中的 `human_player_id` 提交动作：

```bash
curl -X POST http://127.0.0.1:8000/api/matches/MATCH_ID/actions \
  -H 'content-type: application/json' \
  -d '{"actor_id":"PLAYER_ID","action":{"type":"move","payload":{"direction":"right"}}}'
```

单局评价位于 `GET /api/matches/{id}/evaluation`；跨对局实验汇总位于 `GET /api/evaluations/summary`，会按 `MOD:模式` 分组比较综合分与五维分数。

运行评分 v2 的多种子镜像基准（对抗模式会交换双方席位）：

```bash
python -m hva_engine.benchmark --seeds 25
# 安装后也可使用：hva-benchmark --seeds 25
```

## 设计原则

1. **先评价，再扩展**：所有对局从第一回合开始产生同构事件和评分。
2. **MOD 是纯状态机**：规则与展示、传输、Agent 实现解耦。
3. **Agent 只能看到观察值**：为隐藏信息、远程模型与锦标赛留下边界。
4. **直播输入也是动作源**：弹幕与 Godot、Web 控制台共享同一校验链路。
5. **用 MVP 数据升级架构**：评价低分对应明确的下一轮改造方向。

详细内容见 [评价体系](docs/EVALUATION.md)、[架构说明](docs/ARCHITECTURE.md) 与 [LLM/上下文接入](docs/LLM_INTEGRATION.md)。

## 弹幕命令

当前通用入口为 `POST /api/live/danmaku`，接受：

```json
{"match_id":"...", "user":"viewer42", "message":"!move right"}
```

支持的 MVP 命令包括 `!move up`、`!attack`、`!accelerate`、`!conserve`、`!pit`、`!evidence`、`!emotion`、`!rebuttal`。生产环境中可在 `DanmakuAdapter` 前增加 Bilibili、抖音、Twitch 或 YouTube 的鉴权/签名适配器，并保留限流、投票聚合和内容安全层。

## 路线图

- M1：采集真实玩家/Agent 基线，校准评分 v2 权重与难度曲线
- M2：加入并行回合、隐藏信息、回放与持久化
- M3：远程 LLM Agent 沙箱、Elo/Glicko 锦标赛和观众投票窗口
- M4：MOD SDK、资产协议、直播平台正式适配器与 Godot 可视化组件库

## License

Apache-2.0
