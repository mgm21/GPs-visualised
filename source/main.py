from source.core.visualiser import Visualiser

# Fetch some GP models
from agents.example_agents import agents

# Unobserve the points from some GPs
i = 8
agents[i].unobserve_true_points(x=agents[i].x_seen)
agents[i].kappa = 3  # Higher regard for uncertainty works better for toy problem

# Initialise visualiser class
visualiser = Visualiser()

num = 0
# Plot single static GP with all elements
# visualiser.plot_gps_matplotlib(agents[num:num+2], savefig=True)

# # Plot multiple static GPs with some plot elements removed
# visualiser.plot_gps_matplotlib(agents[8:10], plot_elements=["mean", "observed", "var"])

# Start an interactive example app (make sure to give the input as agents[12:13] even just for 1 agent)
# visualiser.visualise_gps_plotly(agents[9:10], plot_elements=["true", "mean", "var", "observed", "acquisition"])

# Start an interactive ITE app (try i = 8 above)
# visualiser.visualise_ite_plotly(agents[i])

# Start an interactive GPCF app (try inputting agents[9:] and include "var" in plot_elements param)
# visualiser.visualise_gpcf_plotly(agents[9:], plot_elements=["mean", "observed"])

# Start an interactive inHERA app
# visualiser.visualise_inhera_plotly(agents[9:], plot_elements=["mean", "observed"])

# # Plot GPs with different elements to plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

plt.style.use("seaborn")
fig, ax = plt.subplots()
# visualiser.update_gps_axes_matplotlib(ax=ax,
#                                       gps_arr=agents[18:19],
#                                       plot_elements=["true", "mean", "observed", "acquisition", "prior",])
#
# visualiser.update_gps_axes_matplotlib(ax=ax,
#                                       gps_arr=agents[19:20],
#                                       plot_elements=["mean",],
#                                       alpha=0.6,
#                                       color="seagreen")
#
# visualiser.update_gps_axes_matplotlib(ax=ax,
#                                       gps_arr=agents[20:21],
#                                       plot_elements=["mean",],
#                                       alpha=0.6,
#                                       color="salmon")

# Manually update the inHERA prior mean function:
from core.gaussian_process import GaussianProcess
gp0 = GaussianProcess(true_func=lambda x: np.sin(0.6 * (x - 5)) + 3 - 1.5 / (x + 0.5),
                mu_0=lambda x: 0.7 * (np.sin(0.6 * (x - 5)) + 3) + 0.3 * (-1.5 * np.cos(0.6 * (x - 9)) + 1 + 0.1 * x),
                x_seen=[1.4])

# 19 Methods/inHERA for report - ANCESTOR 1
gp1 = GaussianProcess(true_func=lambda x: np.sin(0.6 * (x - 5)) + 3,
                mu_0=lambda x: np.sin(0.6 * (x - 5)) + 3,
                x_seen=[1.5, 7.8])

# 20 Methods/inHERA for report - ANCESTOR 2
gp2 = GaussianProcess(true_func=lambda x: -1.5 * np.cos(0.6 * (x - 9)) + 1 + 0.1 * x,
                mu_0=lambda x: -1.5 * np.cos(0.6 * (x - 9)) + 1 + 0.1 * x,
                x_seen=[4])

xplot = gp0.x_problem
a1_mu = gp1.mu_new(xplot)
a1_var = gp1.var_new(xplot)
a2_mu = gp2.mu_new(xplot)
a2_var = gp2.var_new(xplot)

a_vars = np.array([a1_var, a2_var]).T
a_mus = np.array([a1_mu, a2_mu]).T
sim_var = np.full(shape=a_mus.shape, fill_value=1.001)
sim_prior = gp0.mu_0(xplot)

W = np.array([0.7, 0.3])

repeated_sim_mu = np.repeat(a=np.expand_dims(a=sim_prior, axis=1), repeats=2, axis=1)

new_prior = sim_prior + (sim_var - a_vars) * (a_mus - repeated_sim_mu) @ W

gp3 = GaussianProcess(true_func=lambda x: np.sin(0.6 * (x - 5)) + 3 - 1.5 / (x + 0.5),
                mu_0=lambda x: 0.7 * (np.sin(0.6 * (x - 5)) + 3) + 0.3 * (-1.5 * np.cos(0.6 * (x - 9)) + 1 + 0.1 * x),
                x_seen=[1.4])

visualiser.update_gps_axes_matplotlib(ax=ax,
                                      gps_arr=[gp3,])

# family_ancestors_mus =
# print(sim_prior_mean.shape)
# sim_var = np.full(shape=self.family.ancestor_mus.T.shape, fill_value=initial_uncertainty)

# Only change these two lines
font_used = "Charter"
font_size = 21

font = {'fontname': font_used}
legend_font = fm.FontProperties(family=font_used)
legend_font._size = font_size
ax.set_ylim(0, 5)
ax.set_xlim(0, 10)
ax.set_xlabel("Behavioural descriptor", fontsize=font_size, **font)
ax.set_ylabel("Fitness", fontsize=font_size, **font)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
# ax.legend(prop=legend_font,)
fig.savefig("my-plot.png", dpi=600)
plt.show()


