# A Tello SDK "rc a b c d" stick command.
#
# Channels are integers in [-100, 100]. Signs are in the REP-103 body frame used
# throughout this stack, so left_right is positive to the LEFT; the driver flips
# it on the way to the aircraft, whose own channel is positive to the right.
std_msgs/Header header
int8 left_right
int8 forward_back
int8 up_down
int8 yaw_rate
