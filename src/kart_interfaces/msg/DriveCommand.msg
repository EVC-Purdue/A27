# Command from the Pathfinder controller to e_comms
std_msgs/Header header

float64 speed_setpoint  # m/s
float64 steering  # rad, positive left
float64 electrical_brake  # 0 to 1, applied by the VESC
