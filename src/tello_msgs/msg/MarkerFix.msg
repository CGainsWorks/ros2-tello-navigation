# An absolute pose derived from one or more QR markers.
#
# The only non-relative measurement in the system. Everything else says how the
# state changed; this says where the aircraft is.
std_msgs/Header header

geometry_msgs/Point position          # body origin in the map frame, metres
float64 yaw                           # body heading in the map frame, radians

string[] marker_ids                   # map entries that contributed, in order used
uint16 n_points                       # image points in the solve, four per marker
float64 reprojection_rmse             # pixels; the quality signal the gate uses

float64[9] position_covariance        # row-major 3x3, m^2
float64 yaw_variance                  # rad^2
