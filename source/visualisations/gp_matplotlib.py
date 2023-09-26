from source.core.gaussian_process import GaussianProcess
from source.core.visualiser import Visualiser
from source.agents.example_agents import agents

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def configure_adaptation_plot(fig, ax, result_path="my_plot.png", include_legend=True):
    font_used = "Charter"
    font_size = 21
    font = {'fontname': font_used}
    legend_font = fm.FontProperties(family=font_used)
    legend_font._size = 15
    ax.set_ylim(-4, 6)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Behavioural descriptor", fontsize=font_size, **font)
    ax.set_ylabel("Fitness", fontsize=font_size, **font)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    if include_legend:
        ...
        # ax.legend(prop=legend_font, bbox_to_anchor=(1.05, 1), loc='upper right ') # For legend outside the plot
        # ax.legend(prop=legend_font, facecolor="white", frameon=True, ncol=2, loc="upper left") # For legend in the plot
        # ax.legend(prop=legend_font, ncol=2, loc="upper left") # For legend in the plot
    fig.savefig(result_path, dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualiser = Visualiser()
    plt.style.use("seaborn")
    fig1, ax1 = plt.subplots()

    true_func = lambda x: np.sin(x) + 0.5*x
    mu_0 = lambda x: np.sin(x)
    x_seen = [1, 3, 5, 7, 9]
    xplot = np.linspace(start=0, stop=10, num=500)
    yplot = np.linspace(-5, 5, 500)

    # GP instantiation
    gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

    visualiser.update_gps_axes_matplotlib(ax=ax1, gps_arr=[gp], gp_name="", plot_elements=["mean", "prior", "var", "true", "observed"], include_legend=False)
    # visualiser.update_gps_axes_matplotlib(ax=ax1, gps_arr=[agents[6]], plot_elements=["mean"],  gp_name="GP 1", color="seagreen", alpha=0.15)
    # visualiser.update_gps_axes_matplotlib(ax=ax1, gps_arr=[agents[7]], plot_elements=["subtle_var", "mean"],  gp_name="GP 2", color="salmon", alpha=0.15)
    # visualiser.update_gps_axes_matplotlib(ax=ax1, gps_arr=[agents[1]], plot_elements=[ "mean"],  gp_name="GP 3", color="tan", alpha=0.15)

    configure_adaptation_plot(fig1, ax1)
