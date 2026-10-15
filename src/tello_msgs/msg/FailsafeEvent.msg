# One transition of the failsafe state machine, with the reason.
std_msgs/Header header

string from_state
string to_state
string cause                          # one of tello_core.types.FailsafeCause
string detail                         # human-readable, for the flight log
