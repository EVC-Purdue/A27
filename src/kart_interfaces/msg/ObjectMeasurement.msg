# One frame's detection of a kart or obstacle
uint8 CLASS_UNKNOWN=0
uint8 CLASS_KART=1
uint8 CLASS_OBSTACLE=2
uint8 object_class

float64 footprint_size  # m, side of the square footprint
geometry_msgs/Point position  # m
float64[4] position_covariance  # row-major 2x2 over x, y in m^2
