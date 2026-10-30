# One state packet from the aircraft, decoded into SI units and REP-103 axes.
#
# Published by the driver for every valid packet it receives. The decoding, the
# unit conversions and the axis flips happen once, in tello_core.link.state;
# this message carries the result so that the fusion filter and the failsafe
# see exactly what the offline mission runner sees, rather than the subset that
# fits into sensor_msgs/Imu and sensor_msgs/BatteryState.
std_msgs/Header header

geometry_msgs/Vector3 accel           # body frame, m/s^2, specific force
float64 gyro_z                        # rad/s, differentiated from the reported yaw
float64 yaw                           # radians
float64 pitch                         # radians
float64 roll                          # radians
geometry_msgs/Vector3 velocity_body   # body frame, m/s, from the downward optical flow
float64 height                        # metres, downward time of flight
float64 barometer                     # metres
float64 battery                       # fraction, 0 to 1
float64 temperature                   # degrees Celsius
