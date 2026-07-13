extends Control

const API := "http://127.0.0.1:8000"
const TownWorld := preload("res://town_world.gd")

var match_data: Dictionary = {}
var mods: Array = []
var mod_select: OptionButton
var status_label: Label
var clock_label: Label
var weather_label: Label
var action_bar: HFlowContainer
var resident_list: VBoxContainer
var detail_label: RichTextLabel
var event_log: RichTextLabel
var town_world
var request: HTTPRequest
var auto_timer: Timer
var auto_button: Button
var selected_resident_id := ""


func _ready() -> void:
	build_ui()
	request_json("/api/mods", HTTPClient.METHOD_GET, {}, Callable(self, "on_mods"))


func panel_style(color: Color, border: Color, radius := 8) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = color
	style.border_color = border
	style.set_border_width_all(2)
	style.set_corner_radius_all(radius)
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 10
	style.content_margin_bottom = 10
	return style


func build_ui() -> void:
	var background := ColorRect.new()
	background.color = Color("#17202b")
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(background)

	var root := VBoxContainer.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 14)
	root.add_theme_constant_override("separation", 10)
	add_child(root)

	var top := PanelContainer.new()
	top.add_theme_stylebox_override("panel", panel_style(Color("#263444"), Color("#5c7b73")))
	root.add_child(top)
	var top_row := HBoxContainer.new()
	top_row.add_theme_constant_override("separation", 12)
	top.add_child(top_row)
	var title := Label.new()
	title.text = "WILLOW  AGENT  TOWN"
	title.add_theme_color_override("font_color", Color("#f6d58a"))
	title.add_theme_font_size_override("font_size", 24)
	top_row.add_child(title)
	mod_select = OptionButton.new()
	mod_select.custom_minimum_size.x = 250
	top_row.add_child(mod_select)
	var create := Button.new()
	create.text = "生成小镇"
	create.pressed.connect(create_match)
	top_row.add_child(create)
	auto_button = Button.new()
	auto_button.text = "自动观察：关"
	auto_button.toggle_mode = true
	auto_button.toggled.connect(toggle_auto)
	top_row.add_child(auto_button)
	status_label = Label.new()
	status_label.text = "连接后端中…"
	status_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top_row.add_child(status_label)
	clock_label = Label.new()
	clock_label.text = "DAY 1 · 07:00"
	clock_label.add_theme_color_override("font_color", Color("#ffe8a8"))
	top_row.add_child(clock_label)
	weather_label = Label.new()
	weather_label.text = "☀ 晴朗"
	top_row.add_child(weather_label)

	var body := HBoxContainer.new()
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	body.add_theme_constant_override("separation", 10)
	root.add_child(body)

	var world_panel := PanelContainer.new()
	world_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	world_panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	world_panel.clip_contents = true
	world_panel.add_theme_stylebox_override("panel", panel_style(Color("#263a31"), Color("#759b68"), 6))
	body.add_child(world_panel)
	town_world = TownWorld.new()
	town_world.custom_minimum_size = Vector2(720, 500)
	town_world.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	town_world.size_flags_vertical = Control.SIZE_EXPAND_FILL
	town_world.resident_selected.connect(select_resident)
	world_panel.add_child(town_world)

	var sidebar := VBoxContainer.new()
	sidebar.custom_minimum_size.x = 310
	sidebar.add_theme_constant_override("separation", 8)
	body.add_child(sidebar)
	var residents_panel := PanelContainer.new()
	residents_panel.add_theme_stylebox_override("panel", panel_style(Color("#222c3a"), Color("#596f91")))
	residents_panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	sidebar.add_child(residents_panel)
	var residents_content := VBoxContainer.new()
	residents_panel.add_child(residents_content)
	var residents_title := Label.new()
	residents_title.text = "镇民 / 独立 AgentBrain"
	residents_title.add_theme_color_override("font_color", Color("#95d8d0"))
	residents_title.add_theme_font_size_override("font_size", 17)
	residents_content.add_child(residents_title)
	resident_list = VBoxContainer.new()
	resident_list.add_theme_constant_override("separation", 5)
	residents_content.add_child(resident_list)
	detail_label = RichTextLabel.new()
	detail_label.bbcode_enabled = true
	detail_label.custom_minimum_size.y = 175
	detail_label.fit_content = false
	residents_content.add_child(detail_label)

	var log_panel := PanelContainer.new()
	log_panel.add_theme_stylebox_override("panel", panel_style(Color("#2e2933"), Color("#80678d")))
	log_panel.custom_minimum_size.y = 155
	sidebar.add_child(log_panel)
	event_log = RichTextLabel.new()
	event_log.bbcode_enabled = true
	event_log.scroll_following = true
	log_panel.add_child(event_log)

	var bottom := PanelContainer.new()
	bottom.add_theme_stylebox_override("panel", panel_style(Color("#2d3742"), Color("#7f6f4c")))
	root.add_child(bottom)
	action_bar = HFlowContainer.new()
	action_bar.custom_minimum_size.y = 64
	action_bar.add_theme_constant_override("h_separation", 8)
	action_bar.add_theme_constant_override("v_separation", 6)
	bottom.add_child(action_bar)

	request = HTTPRequest.new()
	add_child(request)
	auto_timer = Timer.new()
	auto_timer.wait_time = 1.15
	auto_timer.timeout.connect(auto_step)
	add_child(auto_timer)


func request_json(path: String, method: HTTPClient.Method, body: Dictionary, callback: Callable) -> void:
	if request.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return
	var headers := PackedStringArray(["Content-Type: application/json"])
	request.request_completed.connect(func(_result, code, _headers, bytes):
		var parsed = JSON.parse_string(bytes.get_string_from_utf8())
		callback.call(code, parsed)
	, CONNECT_ONE_SHOT)
	request.request(API + path, headers, method, JSON.stringify(body) if not body.is_empty() else "")


func on_mods(code: int, data) -> void:
	if code != 200 or not (data is Array):
		status_label.text = "后端不可用：请启动 uvicorn hva_engine.api:app --reload"
		return
	mods = data
	var town_index := 0
	for index in range(mods.size()):
		var mod: Dictionary = mods[index]
		mod_select.add_item("%s · %s" % [mod.name, mod.description])
		if mod.id == "agent_town":
			town_index = index
	mod_select.select(town_index)
	status_label.text = "后端已连接 · 选择 Agent 小镇开始"


func create_match() -> void:
	if mods.is_empty():
		return
	var mod_id: String = mods[mod_select.selected].id
	var payload := {"mod_id": mod_id, "human_name": "Observer", "seed": 23}
	if mod_id == "agent_town":
		payload["agent_memory_owner_ids"] = [
			"willow-astra", "willow-nova", "willow-mira"
		]
	status_label.text = "正在生成小镇…"
	request_json("/api/matches", HTTPClient.METHOD_POST, payload, Callable(self, "on_match"))


func on_match(code: int, data) -> void:
	if code < 200 or code >= 300 or not (data is Dictionary):
		status_label.text = "请求失败: %s" % str(data)
		auto_timer.stop()
		return
	match_data = data
	if match_data.mod_id != "agent_town":
		status_label.text = "此 Godot 场景针对 agent_town；其他 MOD 请使用 Web 调试台"
		render_action_bar()
		return
	render_match()


func submit_action(action: Dictionary) -> void:
	if match_data.is_empty() or match_data.status != "active":
		return
	request_json("/api/matches/%s/actions" % match_data.id, HTTPClient.METHOD_POST, {
		"actor_id": match_data.human_player_id,
		"action": action
	}, Callable(self, "on_match"))


func action_label(action: Dictionary) -> String:
	var payload: Dictionary = action.get("payload", {})
	match action.type:
		"wait": return "等待 30 分钟"
		"move_to":
			var location = match_data.state.locations.get(payload.destination, {})
			return "前往 %s" % location.get("name", payload.destination)
		"work": return "进行工作"
		"rest": return "休息"
		"explore": return "四处探索"
		"socialize":
			var target = match_data.state.residents.get(payload.target_id, {})
			return "与 %s 交谈" % target.get("name", "镇民")
		"respond_incident": return "处理突发事件"
		"seek_shelter": return "前往旅店避险"
		"check_bulletin": return "查阅公告与新闻"
		"support_neighbor":
			var neighbor = match_data.state.residents.get(payload.target_id, {})
			return "安慰 %s" % neighbor.get("name", "邻居")
		"check_phone":
			var platform = match_data.state.social_media.platforms.get(payload.platform_id, {})
			return "手机浏览 %s" % platform.get("display_name", "社交媒体")
		"publish_post": return "发布有来源的帖子"
		"reshare_post": return "转发帖子（保留来源）"
		"comment_post": return "评论并讨论"
		"verify_claim": return "求证帖子说法"
		"investigate_claim": return "调查原始证据"
	return str(action.type)


func render_match() -> void:
	var state: Dictionary = match_data.state
	status_label.text = "%s · 回合 %s · %s" % [
		"运行中" if match_data.status == "active" else "三日观察结束",
		state.turn,
		match_data.id.left(8)
	]
	clock_label.text = "DAY %s · %s" % [state.day, state.time]
	var world: Dictionary = state.get("world", {})
	var condition: String = str(world.get("weather", state.weather))
	var weather_icon := "⚡" if "暴雨" in condition else ("☂" if "雨" in condition else "☀")
	weather_label.text = "%s %s · 风险 %d%%" % [
		weather_icon, condition, int(float(world.get("risk_level", 0.0)) * 100.0)
	]
	weather_label.text += " · 帖子 %d" % match_data.state.get("social_media", {}).get("posts", []).size()
	town_world.apply_match(match_data)
	if selected_resident_id.is_empty():
		for player in match_data.players:
			if player.kind == "agent":
				selected_resident_id = player.id
				break
	render_residents()
	render_action_bar()
	render_events()
	if match_data.status != "active":
		auto_timer.stop()
		auto_button.button_pressed = false
		auto_button.text = "自动观察：关"


func render_residents() -> void:
	for child in resident_list.get_children():
		child.queue_free()
	var residents: Dictionary = match_data.state.residents
	for player in match_data.players:
		if player.kind != "agent":
			continue
		var resident: Dictionary = residents[player.id]
		var button := Button.new()
		button.alignment = HORIZONTAL_ALIGNMENT_LEFT
		button.text = "%s  ·  %s\n  ⚡%d  ☺%d  %s" % [
			resident.name,
			resident.job_name,
			int(resident.energy),
			int(resident.mood),
			resident.activity
		]
		button.custom_minimum_size.y = 54
		button.pressed.connect(select_resident.bind(player.id))
		resident_list.add_child(button)
	select_resident(selected_resident_id)


func select_resident(resident_id: String) -> void:
	if match_data.is_empty() or not match_data.state.residents.has(resident_id):
		return
	selected_resident_id = resident_id
	town_world.selected_resident_id = resident_id
	town_world.queue_redraw()
	var resident: Dictionary = match_data.state.residents[resident_id]
	var summary: Dictionary = match_data.agent_summaries.get(resident_id, {})
	var identity: Dictionary = summary.get("identity", {})
	var skills: Dictionary = summary.get("skill_learning", {})
	var stages: Dictionary = skills.get("stages", {})
	var narrative: Dictionary = summary.get("narrative", {})
	var relation_lines := PackedStringArray()
	for target_id in resident.relationships:
		var target = match_data.state.residents.get(target_id, {})
		relation_lines.append("%s %d" % [target.get("name", "?"), int(resident.relationships[target_id])])
	detail_label.text = (
		"[color=#f6d58a][font_size=19]%s[/font_size][/color]  %s\n" % [resident.name, identity.get("archetype", "agent")]
		+ "[color=#9eb8d9]地点[/color] %s   [color=#9eb8d9]活动[/color] %s\n" % [
			match_data.state.locations[resident.location].name, resident.activity]
		+ "[color=#9eb8d9]性格表现[/color] %s\n" % identity.get("social_style", "尚未观察")
		+ "[color=#9eb8d9]工作[/color] %.1f   [color=#9eb8d9]社交[/color] %.1f   [color=#9eb8d9]公民行动[/color] %.1f\n" % [
			resident.work_xp, resident.social_xp, resident.get("civic_xp", 0.0)]
		+ "[color=#9eb8d9]身世揭露[/color] %d/%d\n" % [
			int(narrative.get("revealed_beats", 0)), int(narrative.get("total_beats", 0))]
		+ "[color=#9eb8d9]技能阶段[/color] %s\n" % str(stages)
		+ "[color=#9eb8d9]已自动化[/color] %s\n" % str(skills.get("automatic_skills", []))
		+ "[color=#9eb8d9]关系[/color] %s" % ", ".join(relation_lines)
	)


func render_action_bar() -> void:
	for child in action_bar.get_children():
		child.queue_free()
	if match_data.is_empty():
		return
	for action in match_data.get("legal_actions", []):
		var button := Button.new()
		button.text = action_label(action)
		button.pressed.connect(submit_action.bind(action))
		action_bar.add_child(button)


func render_events() -> void:
	var lines: Array[String] = ["[color=#d7a9e3]小镇事件流[/color]"]
	for event in match_data.events.slice(max(0, match_data.events.size() - 10)):
		if str(event.type).begins_with("town_world_"):
			var severity := int(float(event.payload.get("severity", 0.0)) * 100.0)
			lines.append("[color=#ffcf78]▣ %s[/color]  [风险 %d%%]\n  %s" % [
				event.payload.get("headline", "世界事件"), severity,
				event.payload.get("summary", "")
			])
		elif event.type == "town_incident_response":
			var responder = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("[color=#92dbb7]◆ %s 响应事件，贡献 %.1f[/color]" % [
				responder, event.payload.get("effort", 0.0)
			])
		elif event.type == "town_sheltered":
			var sheltered = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("• %s 前往旅店避险" % sheltered)
		elif event.type == "town_news_checked":
			var reader = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("• %s 核对了公告栏" % reader)
		elif event.type == "town_neighbor_supported":
			var helper = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("• %s 在危机中安慰了邻居" % helper)
		elif event.type == "town_rumor_seeded":
			lines.append("[color=#ff8f8f]⚠ 未经证实的视频开始传播[/color]")
		elif event.type == "town_social_posted":
			var author = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("[color=#8ed3f4]▤ %s 发帖[/color]：%s" % [
				author, event.payload.get("content", "")
			])
		elif event.type == "town_social_reshared":
			var sharer = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("• %s 转发帖子 · 失真 %.0f%%" % [
				sharer, float(event.payload.get("distortion", 0.0)) * 100.0
			])
		elif event.type == "town_social_commented":
			var commenter = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("• %s 评论：%s" % [commenter, event.payload.get("content", "")])
		elif event.type == "town_claim_verified":
			var verifier = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("[color=#91dbac]✓ %s 求证：%s[/color]" % [
				verifier, event.payload.get("result", "unresolved")
			])
		elif event.type == "town_claim_investigated":
			var investigator = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("[color=#e6c96a]⌕ %s 调查：%s[/color]" % [
				investigator, event.payload.get("finding", "获得新证据")
			])
		elif event.type == "town_conversation":
			var speaker = match_data.state.residents.get(event.actor_id, {}).get("name", "Agent")
			lines.append("[color=#f2c6a0]%s[/color]：%s" % [speaker, event.payload.dialogue])
		elif event.type == "town_moved":
			var mover = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			var place = match_data.state.locations.get(event.payload.destination, {}).get("name", "远方")
			lines.append("• %s 前往%s" % [mover, place])
		elif event.type == "town_worked":
			var worker = match_data.state.residents.get(event.actor_id, {}).get("name", "镇民")
			lines.append("• %s 完成了 %.1f 工作进度" % [worker, event.payload.progress])
	event_log.text = "\n".join(lines)


func toggle_auto(enabled: bool) -> void:
	auto_button.text = "自动观察：开" if enabled else "自动观察：关"
	if enabled:
		auto_timer.start()
	else:
		auto_timer.stop()


func auto_step() -> void:
	if match_data.is_empty() or match_data.status != "active":
		return
	if request.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return
	var wait_action = match_data.legal_actions.filter(func(action): return action.type == "wait")
	if not wait_action.is_empty():
		submit_action(wait_action[0])
