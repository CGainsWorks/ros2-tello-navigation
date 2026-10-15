# Health of the link itself, as distinct from the health of the aircraft.
#
# Kept separate on purpose: a link that has heard nothing for two seconds and a
# battery at twelve per cent are different problems with different responses.
std_msgs/Header header

bool connected
float64 last_state_age                # seconds since the last telemetry packet
float64 last_frame_age                # seconds since the last video frame
uint32 packets_received
uint32 packets_dropped
uint32 commands_sent
string detail
