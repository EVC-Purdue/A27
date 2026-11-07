# UM982 fix projected into the local track frame
std_msgs/Header header

uint8 FIX_NONE=0
uint8 FIX_SINGLE=1
uint8 FIX_DGPS=2
uint8 FIX_RTK_FIXED=4
uint8 FIX_RTK_FLOAT=5
# FIX_NONE means x and y carry no position
uint8 fix_quality

float64 x  # m
float64 y  # m
float64 heading  # rad, ENU, valid only when heading_valid
bool heading_valid
float64 speed  # m/s

float64[4] position_covariance  # row-major 2x2 over x, y in m^2
float64 heading_variance  # rad^2
float64 speed_variance  # (m/s)^2
