# Feedback from e_comms
std_msgs/Header header

float64 wheel_speed  # m/s
float64 steering_angle  # rad, positive left
float64 steering_motor_current  # A
float64 motor_current  # A

uint8 RC_MODE_AUTONOMOUS=0
uint8 RC_MODE_OVERRIDE=1
uint8 rc_mode

uint8 board_state
