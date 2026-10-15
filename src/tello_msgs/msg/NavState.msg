# The fused estimate the controller flies on.
std_msgs/Header header

geometry_msgs/Point position          # map frame, metres
geometry_msgs/Vector3 velocity        # map frame, m/s
float64 yaw                           # map frame, radians

geometry_msgs/Vector3 accel_bias      # body frame, m/s^2
float64 gyro_bias                     # rad/s

float64[9] position_covariance        # row-major 3x3, m^2
float64 position_sigma                # scalar summary, metres

bool localised                        # a recent fix and a covariance small enough to act on
float64 time_since_fix                # seconds
