from source.core.gaussian_process import GaussianProcess
from source.core.visualiser import Visualiser
from source.agents.example_agents import agents
from source.visualisations.gp_matplotlib import configure_adaptation_plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Define the gp
agent = GaussianProcess(true_func=np.sin,
                     mu_0=np.cos,
                     x_seen=[],
                     x_problem=np.linspace(start=0, stop=10, num=100),)

true_ancest = GaussianProcess(true_func=np.sin,
                     mu_0=np.sin,
                     x_seen=[0.1, 6.4,],
                     x_problem=np.linspace(start=0, stop=10, num=100),)

false_ancest = GaussianProcess(true_func=np.cos,
                     mu_0=np.cos,
                     x_seen=[0.1, 6.4,],
                     x_problem=np.linspace(start=0, stop=10, num=100),)

visualiser = Visualiser()

# Copied from gp_matplotlib
plt.style.use("seaborn")
fig, ax = plt.subplots()
# Plot the agents
visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent, ], )
visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[true_ancest, ], color="tan", plot_elements="mean")
visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[false_ancest, ], color="olive", plot_elements="mean")
configure_adaptation_plot(fig, ax, result_path=f"gpcf.png")

# Artificial weights (little by little understands who it most like) for visualisation
W_artificial = [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]

for i in range(5):
    x_acquisition = agent.query_acquisition_function()
    agent.observe_true_points(x_acquisition)

    W = W_artificial[i]
    agent.mu_0 = lambda x: W @ np.array([true_ancest.mu_new(x), false_ancest.mu_new(x)])

    # Copied from gp_matplotlib
    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    # Plot the agents
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent,],)
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[true_ancest,], color="tan", plot_elements="mean")
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[false_ancest,], color="olive", plot_elements="mean")
    configure_adaptation_plot(fig, ax, result_path=f"gpcf{i}.png")


# TODO: implement likelihood calculation to get optimal W, for now st set it yourself
#  this is enough for visualising in your report. Can automise for presentation.