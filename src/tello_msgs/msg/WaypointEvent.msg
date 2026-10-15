# A waypoint reached, timed out or abandoned.
std_msgs/Header header

uint16 index
string name
string kind                           # "arrived", "timeout" or "abandoned"
float64 position_error                # metres between the estimate and the waypoint
