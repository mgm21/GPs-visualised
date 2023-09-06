from source.core.gaussian_process import GaussianProcess
from source.core.visualiser import Visualiser
from source.agents.example_agents import agents
from source.visualisations.gp_matplotlib import configure_adaptation_plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Define GPs for the problem
agent = agents[18]
true_ancest = agents[19]
false_ancest = agents[20]
ancestors = np.array([true_ancest, false_ancest])
W_artificial = [[0.7, 0.3], [0.9, 0.1]] # Hand-set weights for now, no max likelihood calculation

# Define the algorithm variables
base = lambda x: np.zeros(shape=x.shape)
var0 = agent._kernel_func(0,0) + agent.sampling_noise
var0_mat = lambda x: np.repeat(a=np.array([np.repeat(a=np.array(var0), repeats=len(ancestors))]), repeats=len(x), axis=0)
ancest_var = lambda x: np.array([true_ancest.var_new(x), false_ancest.var_new(x)]).T
ancest_mean = lambda x: np.array([true_ancest.mu_new(x), false_ancest.mu_new(x)]).T
base_mat = lambda x: np.repeat(a=np.expand_dims(a=base(x), axis=1), repeats=len(ancestors), axis=1)

# Define a visualiser environment
visualiser = Visualiser()
plt.style.use("seaborn")

for i in range(2):
    # Visualise the state of the GP
    fig, ax = plt.subplots()
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent,], gp_name="Child", plot_elements=["prior", "observed", "mean", "var"], alpha=0.1, include_legend=True)
    # visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[agent, ], gp_name="Child", plot_elements=["true"], alpha=0.1, include_legend=False)
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[true_ancest,], color="seagreen", plot_elements=["mean", "subtle_var"], gp_name="Ancestor 1", alpha=0.1, include_legend=False)
    visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=[false_ancest,], color="salmon", plot_elements=["mean", "subtle_var"], gp_name="Ancestor 2", alpha=0.1, include_legend=False)
    configure_adaptation_plot(fig, ax, result_path=f"inhera{i}.png", include_legend=True)

    x_acquisition = agent.query_acquisition_function()
    # agent.observe_true_points(x_acquisition)

    W = W_artificial[i]
    agent.mu_0 = lambda x: base(x) + (var0_mat(x) - ancest_var(x)) * (ancest_mean(x) - base_mat(x)) @ W