// SPDX-FileCopyrightText: James Webb
// SPDX-License-Identifier: MIT
// Full details are in the LICENSE file at the repository root.

#import "Basic";
#import "Math";

Camera :: struct
{
	FLY_SPEED :: 25.0;
	ROTATION_SPEED :: 0.3;
	ROTATION_SMOOTHING :: 0.3;

	position : Vector3;
	euler_angles : Vector3;
	rotation : Quaternion;
	rotation_input : Vector2;
	fov := 70.0;
	transform : Matrix4 = Matrix4_Identity;
	view_matrix : Matrix4 = Matrix4_Identity;
	projection_matrix : Matrix4 = Matrix4_Identity;
	view_projection_matrix : Matrix4 = Matrix4_Identity;
}

update_camera :: (camera: *Camera, delta_time: float, window_w : s32, window_h: s32)
{
	mouse_delta : Vector2;
	move_input  : Vector3;
	move_speed  : float;

	io := ImGui.GetIO();
	accept_input :=	!ImGui.IsAnyItemActive() && !ImGui.IsAnyItemHovered() && !ImGui.IsWindowHovered(.AnyWindow);

	if accept_input {
		move_speed = camera.FLY_SPEED;
		if (ImGui.IsMouseDown(.Left)) {
			mouse_delta = xy (xx io.MouseDelta.x, xx io.MouseDelta.y);
		}
		if ImGui.IsKeyDown(.Space) then move_speed *= 2;
		move_speed *= delta_time;
		move_input.x = cast (float) ImGui.IsKeyDown(.D)
			- cast (float) ImGui.IsKeyDown(.A);
		move_input.y = cast (float) ImGui.IsKeyDown(.E)
			- cast (float) ImGui.IsKeyDown(.Q);
		move_input.z = cast (float) ImGui.IsKeyDown(.S)
			- cast (float) ImGui.IsKeyDown(.W);
		move_input = rotate (unit_vector (move_input), camera.rotation);
	}

    camera.position += move_input * move_speed;    
	camera.rotation_input = lerp (camera.rotation_input, mouse_delta, camera.ROTATION_SMOOTHING);
	delta := camera.rotation_input * camera.ROTATION_SPEED;
	camera.euler_angles.x += delta.x * PI / 180.0;
	camera.euler_angles.y += delta.y * PI / 180.0;
	camera.euler_angles.y = clamp (camera.euler_angles.y, -80.0  * PI / 180.0, 80.0 * PI / 180.0);
	yaw : Quaternion;
	set_from_axis_and_angle (*yaw, 0, -1, 0, camera.euler_angles.x);
	pitch : Quaternion;
	set_from_axis_and_angle (*pitch, -1, 0, 0, camera.euler_angles.y);
	camera.rotation = yaw * pitch;
	camera.transform = make_translation_matrix4 (camera.position) * rotation_matrix (Matrix4, camera.rotation);

    success:, camera.view_matrix = inverse (camera.transform);
	aspect_ratio := window_w / cast (float) window_h;
	camera.projection_matrix = make_projection_matrix (camera.fov * PI / 180.0, aspect_ratio, 0.01, 100.0);
	camera.view_projection_matrix = camera.projection_matrix * camera.view_matrix;
}