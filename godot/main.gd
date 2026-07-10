extends Control

const API := "http://127.0.0.1:8000"
var match_data: Dictionary = {}
var mods: Array = []
var mod_select: OptionButton
var state_view: RichTextLabel
var status_label: Label
var actions: HFlowContainer
var request: HTTPRequest

func _ready() -> void:
	build_ui()
	request_json("/api/mods", HTTPClient.METHOD_GET, {}, Callable(self, "on_mods"))

func build_ui() -> void:
	var root := VBoxContainer.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 26)
	add_child(root)
	var title := Label.new()
	title.text = "HUMAN  VS  AGENT"
	title.add_theme_font_size_override("font_size", 34)
	root.add_child(title)
	var controls := HBoxContainer.new()
	root.add_child(controls)
	mod_select = OptionButton.new()
	mod_select.custom_minimum_size.x = 280
	controls.add_child(mod_select)
	var create := Button.new()
	create.text = "创建对局"
	create.pressed.connect(create_match)
	controls.add_child(create)
	status_label = Label.new()
	status_label.text = "连接后端中…"
	controls.add_child(status_label)
	actions = HFlowContainer.new()
	actions.custom_minimum_size.y = 70
	root.add_child(actions)
	state_view = RichTextLabel.new()
	state_view.bbcode_enabled = true
	state_view.fit_content = false
	state_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(state_view)
	request = HTTPRequest.new()
	add_child(request)

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
	if code != 200:
		status_label.text = "后端不可用，请启动 uvicorn"
		return
	mods = data
	for mod in mods:
		mod_select.add_item("%s · %s" % [mod.name, mod.description])
	status_label.text = "后端已连接"

func create_match() -> void:
	if mods.is_empty(): return
	request_json("/api/matches", HTTPClient.METHOD_POST, {
		"mod_id": mods[mod_select.selected].id,
		"human_name": "Godot Player"
	}, Callable(self, "on_match"))

func on_match(code: int, data) -> void:
	if code < 200 or code >= 300:
		status_label.text = "请求失败: %s" % str(data)
		return
	match_data = data
	render_match()

func submit_action(action: Dictionary) -> void:
	request_json("/api/matches/%s/actions" % match_data.id, HTTPClient.METHOD_POST, {
		"actor_id": match_data.human_player_id,
		"action": action
	}, Callable(self, "on_match"))

func render_match() -> void:
	status_label.text = "%s · %s" % [match_data.status, match_data.id.left(8)]
	for child in actions.get_children(): child.queue_free()
	for action in match_data.legal_actions:
		var button := Button.new()
		button.text = str(action.type)
		button.pressed.connect(submit_action.bind(action))
		actions.add_child(button)
	state_view.text = "[color=#58e6d9]STATE[/color]\n%s\n\n[color=#ff5c7a]SCORES[/color]\n%s\n\nAGENTS\n%s" % [
		JSON.stringify(match_data.state, "  "), JSON.stringify(match_data.scores, "  "),
		JSON.stringify(match_data.agent_summaries, "  ")
	]
