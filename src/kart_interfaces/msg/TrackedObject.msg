# One tracked kart or obstacle
uint32 id

uint8 CLASS_UNKNOWN=0
uint8 CLASS_KART=1
uint8 CLASS_OBSTACLE=2
uint8 object_class

float64 footprint_size  # m, side of the square footprint
geometry_msgs/Point position  # m
geometry_msgs/Vector3 velocity  # m/s
float64[16] covariance  # row-major 4x4 over x, y, vx, vy
float64 age  # s since the track was created
