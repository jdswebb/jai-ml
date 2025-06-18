#import "Basic";
#import "Math";

Camera :: struct
{
	FLY_SPEED :: 1.0;
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

is_key_down :: inline (key : Key_Code) -> bool
{
	return (input_button_states[cast (s64) key] & .DOWN) != 0;
}

update_camera :: (camera: *Camera, delta_time: float) {
	mouse_delta : Vector2;
	move_input  : Vector3;
	move_speed  : float;

	if input_button_states[Key_Code.MOUSE_BUTTON_LEFT] {
		move_speed = camera.FLY_SPEED;
		mouse_delta = xy (xx mouse_delta_x, xx mouse_delta_y);
		if is_key_down (.SHIFT) then move_speed *= 10;
		move_speed *= delta_time;
		move_input.x = cast (float) is_key_down (xx #char "D")
			- cast (float) is_key_down (xx #char "A");
		move_input.y = cast (float) is_key_down (xx #char "E")
			- cast (float) is_key_down (xx #char "Q");
		move_input.z = cast (float) is_key_down (xx #char "S")
			- cast (float) is_key_down (xx #char "W");
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
	vx, vy, vw, vh := get_dimensions (g_window, false);
	aspect_ratio := vw / cast (float) vh;
	camera.projection_matrix = make_projection_matrix (camera.fov * PI / 180.0, aspect_ratio, 0.01, 100.0);
	camera.view_projection_matrix = camera.projection_matrix * camera.view_matrix;
}