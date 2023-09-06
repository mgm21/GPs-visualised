from source.core.gaussian_process import GaussianProcess
from source.core.visualiser import Visualiser
from source.agents.example_agents import agents
from source.visualisations.gp_matplotlib import configure_adaptation_plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Define the gp
agent = agents[14]

visualiser = Visualiser()

for i in range(5):
    plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=(10, 6))
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent],)
    configure_adaptation_plot(fig, ax, result_path=f"ite_{i}.png", include_legend=True)

    x_acquisition = agent.query_acquisition_function()
    agent.observe_true_points(x_acquisition)


