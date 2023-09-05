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

visualiser = Visualiser()

for i in range(5):
    x_acquisition = agent.query_acquisition_function()
    agent.observe_true_points(x_acquisition)

    # Copied from gp_matplotlib
    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent],)
    configure_adaptation_plot(fig, ax, result_path=f"ite_{i}.png")


