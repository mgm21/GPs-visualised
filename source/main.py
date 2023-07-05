from source.core.visualiser import Visualiser

# Fetch some optimised GPs
from agents.example_agents import agents

# Unobserve the points from some GPs
agents[1].unobserve_true_points(x=agents[1].x_seen)
print(agents[1].x_seen)

# Launch the visualisation app
visualiser = Visualiser()
visualiser.visualise_ite_plotly(agents[1])
visualiser.visualise_gpcf_plotly(agents[:3])

# Backlog:

# Todo: add more ancestors: multiple whose maxes are in the middle (offset the sine by a bit) AND multiple ones who have
#  the same defect (left side down or right side down) so that it can learn well from ancestors.

# Todo: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)
