from source.core.visualiser import Visualiser

# Fetch some optimised GPs
from agents.example_agents import agents

# Unobserve the points from some GPs
i = 8
agents[i].unobserve_true_points(x=agents[i].x_seen)
agents[i].kappa = 3  # Higher regard for uncertainty works better for toy problem

# Initialise visualiser class
visualiser = Visualiser()

# Plot single static GP with all elements
# visualiser.plot_gps_matplotlib(agents[9:10])

# Plot multiple static GPs with some plot elements removed
# visualiser.plot_gps_matplotlib(agents[9:], plot_elements=["mean", "observed"])

# Start an interactive example app
# visualiser.visualise_gps_plotly(agents[1:3], plot_elements=["mean", "var"])

# Start an interactive ITE app (try i = 8 above)
# visualiser.visualise_ite_plotly(agents[i])

# Start an interactive GPCF app (try inputting agents[9:] and include "var" in plot_elements param)
# visualiser.visualise_gpcf_plotly(agents[9:], plot_elements=["mean", "observed"])

# Start an interactive inHERA app
# visualiser.visualise_inhera_plotly(agents[9:], plot_elements=["mean", "observed"])
