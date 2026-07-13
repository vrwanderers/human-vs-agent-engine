extends Control

signal resident_selected(resident_id: String)

const MAP_SIZE := Vector2(920.0, 600.0)
const AGENT_COLORS := [
	Color("#e96f6f"), Color("#6fa8e9"), Color("#d98bea"), Color("#f0b45d"),
	Color("#65c6a2"), Color("#ddd06a"), Color("#a998ef"), Color("#ec91b5")
]

var match_data: Dictionary = {}
var display_positions: Dictionary = {}
var target_positions: Dictionary = {}
var color_by_id: Dictionary = {}
var selected_resident_id := ""
var hover_resident_id := ""


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	set_process(true)


func apply_match(data: Dictionary) -> void:
	match_data = data
	var agent_index := 0
	for player in data.players:
		var resident: Dictionary = data.state.residents[player.id]
		var target := Vector2(float(resident.x), float(resident.y))
		target_positions[player.id] = target
		if not display_positions.has(player.id):
			display_positions[player.id] = target
		if player.kind == "human":
			color_by_id[player.id] = Color("#f5f1e8")
		else:
			color_by_id[player.id] = AGENT_COLORS[agent_index % AGENT_COLORS.size()]
			agent_index += 1
	queue_redraw()


func _process(delta: float) -> void:
	var changed := false
	for resident_id in target_positions:
		var before: Vector2 = display_positions[resident_id]
		var after: Vector2 = before.lerp(target_positions[resident_id], min(1.0, delta * 3.4))
		if before.distance_to(after) > 0.05:
			display_positions[resident_id] = after
			changed = true
	if changed:
		queue_redraw()
	elif not match_data.is_empty():
		var world: Dictionary = match_data.state.get("world", {})
		var has_incident := not world.get("active_incidents", []).is_empty()
		if has_incident or "暴雨" in str(world.get("weather", "")):
			queue_redraw()


func map_transform() -> Dictionary:
	var available := size - Vector2(18, 18)
	var scale_value: float = min(available.x / MAP_SIZE.x, available.y / MAP_SIZE.y)
	var drawn := MAP_SIZE * scale_value
	return {"scale": scale_value, "origin": (size - drawn) * 0.5}


func point(value: Vector2) -> Vector2:
	var transform := map_transform()
	return transform.origin + value * transform.scale


func scaled_rect(rect: Rect2) -> Rect2:
	var transform := map_transform()
	return Rect2(transform.origin + rect.position * transform.scale, rect.size * transform.scale)


func _draw() -> void:
	var map_rect := scaled_rect(Rect2(Vector2.ZERO, MAP_SIZE))
	draw_rect(map_rect, Color("#78ad5a"))
	draw_rect(scaled_rect(Rect2(0, 0, 920, 48)), Color("#5f914d"))
	draw_rect(scaled_rect(Rect2(0, 552, 920, 48)), Color("#5f914d"))

	# Dirt paths join every daily destination through the square.
	var square := point(Vector2(455, 295))
	for destination in [Vector2(150, 150), Vector2(765, 150), Vector2(770, 430), Vector2(155, 430), Vector2(460, 520)]:
		draw_line(square, point(destination), Color("#d3bd83"), max(8.0, map_transform().scale * 24.0), true)
		draw_line(square, point(destination), Color("#e5d29c"), max(3.0, map_transform().scale * 13.0), true)

	# Farm plots, lake and plaza form recognizable gameplay landmarks.
	for row in 3:
		for column in 5:
			draw_rect(scaled_rect(Rect2(45 + column * 32, 205 + row * 20, 25, 10)), Color("#73553a"))
			draw_line(point(Vector2(48 + column * 32, 210 + row * 20)), point(Vector2(66 + column * 32, 210 + row * 20)), Color("#91c65c"), 2)
	draw_circle(point(Vector2(155, 430)), map_transform().scale * 67, Color("#4b91a4"))
	draw_circle(point(Vector2(155, 430)), map_transform().scale * 52, Color("#65b4c4"))
	draw_circle(square, map_transform().scale * 42, Color("#d5c59d"))
	draw_circle(square, map_transform().scale * 12, Color("#8ab0bd"))

	# Pixel-like buildings. They are code-native so map coordinates remain authoritative.
	draw_building(Vector2(150, 150), Vector2(104, 70), Color("#c7804e"), Color("#7d493c"), "晨露农场")
	draw_building(Vector2(765, 150), Vector2(112, 76), Color("#a99a8d"), Color("#5c5963"), "齿轮工坊")
	draw_building(Vector2(770, 430), Vector2(118, 82), Color("#c8ab70"), Color("#526f65"), "月桂图书馆")
	draw_building(Vector2(460, 520), Vector2(128, 78), Color("#d69a62"), Color("#704b55"), "橡果旅店")
	draw_label(Vector2(455, 345), "风铃广场", Color("#334637"))
	draw_label(Vector2(155, 510), "镜湖", Color("#eaf7ee"))

	for tree_position in [Vector2(270, 90), Vector2(350, 105), Vector2(565, 90), Vector2(650, 95), Vector2(270, 470), Vector2(650, 500), Vector2(85, 350), Vector2(850, 325)]:
		draw_tree(tree_position)
	for flower_position in [Vector2(330, 250), Vector2(375, 370), Vector2(560, 375), Vector2(605, 255), Vector2(240, 330)]:
		draw_flower(flower_position)

	if match_data.is_empty():
		draw_label(Vector2(460, 300), "生成小镇后，Agent 会在这里生活", Color("#f6f0cf"), 20)
		return
	draw_world_conditions()
	var offsets: Dictionary = {}
	for player in match_data.players:
		var resident: Dictionary = match_data.state.residents[player.id]
		var location_id: String = resident.location
		var count: int = offsets.get(location_id, 0)
		offsets[location_id] = count + 1
		var base_position: Vector2 = display_positions.get(player.id, Vector2(resident.x, resident.y))
		var offset := Vector2((count % 3 - 1) * 20, floor(count / 3.0) * 22 + 22)
		draw_resident(player.id, base_position + offset, resident, player.kind == "human")


func draw_world_conditions() -> void:
	var world: Dictionary = match_data.state.get("world", {})
	var condition := str(world.get("weather", match_data.state.get("weather", "")))
	if "暴雨" in condition:
		var map_rect := scaled_rect(Rect2(Vector2.ZERO, MAP_SIZE))
		draw_rect(map_rect, Color(0.08, 0.14, 0.23, 0.24))
		var shift := float(Time.get_ticks_msec() % 700) / 700.0 * 22.0
		for index in 42:
			var x := float((index * 83) % 910) + 5.0
			var y := fmod(float(index * 47) + shift, 570.0) + 12.0
			draw_line(point(Vector2(x, y)), point(Vector2(x - 8, y + 18)), Color(0.72, 0.86, 1.0, 0.65), max(1.0, map_transform().scale * 1.5))
	var pulse := 1.0 + 0.15 * sin(float(Time.get_ticks_msec()) / 170.0)
	for incident in world.get("active_incidents", []):
		if incident.get("status", "") != "active":
			continue
		for location_id in incident.get("affected_locations", []):
			var location: Dictionary = match_data.state.locations.get(location_id, {})
			if location.is_empty():
				continue
			var center := point(Vector2(float(location.x), float(location.y)))
			draw_arc(center, map_transform().scale * 52.0 * pulse, 0, TAU, 32, Color("#ff675f"), max(2.0, map_transform().scale * 4.0))
			draw_string(ThemeDB.fallback_font, center + Vector2(-7, -38), "!", HORIZONTAL_ALIGNMENT_CENTER, 18, 24, Color("#ffe17c"))


func draw_building(center: Vector2, building_size: Vector2, wall: Color, roof: Color, label: String) -> void:
	var body := Rect2(center - Vector2(building_size.x * 0.5, building_size.y * 0.25), building_size)
	draw_rect(scaled_rect(body), wall)
	var roof_points := PackedVector2Array([
		point(center + Vector2(-building_size.x * 0.62, -building_size.y * 0.22)),
		point(center + Vector2(0, -building_size.y * 0.72)),
		point(center + Vector2(building_size.x * 0.62, -building_size.y * 0.22))
	])
	draw_colored_polygon(roof_points, roof)
	draw_rect(scaled_rect(Rect2(center + Vector2(-10, building_size.y * 0.28), Vector2(20, building_size.y * 0.47))), Color("#493b35"))
	draw_rect(scaled_rect(Rect2(center + Vector2(-building_size.x * 0.36, 2), Vector2(18, 16))), Color("#9bd0da"))
	draw_label(center + Vector2(0, building_size.y * 0.8), label, Color("#f7efd2"))


func draw_tree(tree_position: Vector2) -> void:
	draw_rect(scaled_rect(Rect2(tree_position + Vector2(-4, 6), Vector2(8, 22))), Color("#6d4933"))
	draw_circle(point(tree_position), map_transform().scale * 18, Color("#315f3e"))
	draw_circle(point(tree_position + Vector2(-9, -3)), map_transform().scale * 12, Color("#477d48"))
	draw_circle(point(tree_position + Vector2(9, -2)), map_transform().scale * 12, Color("#57944f"))


func draw_flower(flower_position: Vector2) -> void:
	var center := point(flower_position)
	draw_circle(center, max(2.0, map_transform().scale * 3), Color("#f8d477"))
	for direction in [Vector2.LEFT, Vector2.RIGHT, Vector2.UP, Vector2.DOWN]:
		draw_circle(center + direction * max(3.0, map_transform().scale * 4), max(2.0, map_transform().scale * 3), Color("#f2a6bf"))


func draw_resident(resident_id: String, world_position: Vector2, resident: Dictionary, is_human: bool) -> void:
	var screen := point(world_position)
	var scale_value: float = map_transform().scale
	var body_color: Color = color_by_id.get(resident_id, Color.WHITE)
	draw_ellipse_shadow(screen + Vector2(0, 12 * scale_value), Vector2(13, 5) * scale_value)
	draw_rect(Rect2(screen + Vector2(-7, -1) * scale_value, Vector2(14, 18) * scale_value), body_color)
	draw_rect(Rect2(screen + Vector2(-6, 15) * scale_value, Vector2(5, 7) * scale_value), Color("#3f4550"))
	draw_rect(Rect2(screen + Vector2(1, 15) * scale_value, Vector2(5, 7) * scale_value), Color("#3f4550"))
	draw_rect(Rect2(screen + Vector2(-8, -13) * scale_value, Vector2(16, 14) * scale_value), Color("#f0c49a"))
	draw_rect(Rect2(screen + Vector2(-8, -15) * scale_value, Vector2(16, 5) * scale_value), body_color.darkened(0.35))
	draw_rect(Rect2(screen + Vector2(-4, -7) * scale_value, Vector2(2, 2) * scale_value), Color("#26313b"))
	draw_rect(Rect2(screen + Vector2(3, -7) * scale_value, Vector2(2, 2) * scale_value), Color("#26313b"))
	if resident_id == selected_resident_id:
		draw_arc(screen, 18 * scale_value, 0, TAU, 24, Color("#ffe288"), max(2.0, 3 * scale_value))
	if is_human:
		draw_arc(screen, 21 * scale_value, PI, TAU, 12, Color("#ffffff"), max(1.0, 2 * scale_value))
	draw_label(world_position + Vector2(0, 36), resident.name, Color("#fff8dc"), 13)
	if resident_id == hover_resident_id or resident_id == selected_resident_id:
		draw_bubble(world_position + Vector2(0, -52), str(resident.activity))


func draw_ellipse_shadow(center: Vector2, radius: Vector2) -> void:
	var points := PackedVector2Array()
	for index in 16:
		var angle := TAU * index / 16.0
		points.append(center + Vector2(cos(angle) * radius.x, sin(angle) * radius.y))
	draw_colored_polygon(points, Color(0.1, 0.15, 0.12, 0.35))


func draw_label(world_position: Vector2, text: String, color: Color, font_size := 14) -> void:
	var screen := point(world_position)
	var width := max(80.0, text.length() * font_size * 0.85)
	draw_string(ThemeDB.fallback_font, screen - Vector2(width * 0.5, 0), text, HORIZONTAL_ALIGNMENT_CENTER, width, font_size, color)


func draw_bubble(world_position: Vector2, text: String) -> void:
	var compact := text.left(18)
	var screen := point(world_position)
	var width := max(92.0, compact.length() * 12.0)
	var bubble := Rect2(screen - Vector2(width * 0.5, 18), Vector2(width, 25))
	draw_rect(bubble, Color("#fff6d8"))
	draw_rect(bubble, Color("#5c4d43"), false, 2)
	draw_string(ThemeDB.fallback_font, bubble.position + Vector2(6, 17), compact, HORIZONTAL_ALIGNMENT_LEFT, width - 12, 12, Color("#3c332d"))


func _gui_input(event: InputEvent) -> void:
	if match_data.is_empty():
		return
	if event is InputEventMouseMotion:
		hover_resident_id = nearest_resident(event.position)
		queue_redraw()
	elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		var selected := nearest_resident(event.position)
		if not selected.is_empty():
			resident_selected.emit(selected)


func nearest_resident(screen_position: Vector2) -> String:
	var nearest := ""
	var distance := 28.0
	for resident_id in display_positions:
		var candidate_distance: float = screen_position.distance_to(point(display_positions[resident_id]))
		if candidate_distance < distance:
			distance = candidate_distance
			nearest = resident_id
	return nearest
