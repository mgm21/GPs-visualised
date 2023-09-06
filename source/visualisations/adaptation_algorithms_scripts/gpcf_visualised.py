from source.core.gaussian_process import GaussianProcess
from source.core.visualiser import Visualiser
from source.agents.example_agents import agents
from source.visualisations.gp_matplotlib import configure_adaptation_plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Define the gp
agent = agents[15]
true_ancest = agents[16]
false_ancest = agents[17]

W_artificial = [[0.5, 0.5],]

visualiser = Visualiser()
plt.style.use("seaborn")

for i in range(2):
    fig, ax = plt.subplots()
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent,], gp_name="Child")
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[true_ancest,], color="seagreen", plot_elements="mean", gp_name="Ancestor 1", include_legend=False)
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[false_ancest,], color="salmon", plot_elements="mean", gp_name="Ancestor 2", include_legend=False)
    configure_adaptation_plot(fig, ax, result_path=f"gpcf{i}.png", include_legend=True)

    x_acquisition = agent.query_acquisition_function()
    agent.observe_true_points(x_acquisition)

    W = W_artificial[i]
    agent.mu_0 = lambda x: W @ np.array([true_ancest.mu_new(x), false_ancest.mu_new(x)])