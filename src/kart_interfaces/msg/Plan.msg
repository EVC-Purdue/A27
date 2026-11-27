# Path the controller is following ahead of the kart, with its costs when the controller computes them
std_msgs/Header header
LinePoint[] points
string[] cost_terms  # names of the cost terms
float64[] step_costs  # for each step between points in turn, one value per cost term
float64 best_cost  # total cost of the plan
float64 cost_field_start  # m, offset to the side of the line of the first cost_field value
float64 cost_field_step  # m, offset between cost_field values
float64[] cost_field  # position cost at each offset
