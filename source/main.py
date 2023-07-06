from source.core.visualiser import Visualiser

# Fetch some optimised GPs
from agents.example_agents import agents

# Unobserve the points from some GPs
i = 8
agents[i].unobserve_true_points(x=agents[i].x_seen)
# Higher regard for uncertainty works better for toy problem
# not to get stuck in local maxima
agents[i].kappa = 3

# Launch the visualisation app
visualiser = Visualiser()
visualiser.plot_gps_matplotlib(agents[9:], plot_elements=["mean", "var"])
# visualiser.visualise_ite_plotly(agents[i])
# visualiser.visualise_gpcf_plotly(agents[:3])

# Backlog:

# Todo: add more ancestors: multiple whose maxes are in the middle (offset the sine by a bit) AND multiple ones who have
#  the same defect (left side down or right side down) so that it can learn well from ancestors.

# Todo: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)
