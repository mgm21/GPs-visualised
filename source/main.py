from gaussian_process import GaussianProcess
from visualiser import Visualiser
import numpy as np

# DEFINE THE GPs
agents = [
    # Intact ancestor
    GaussianProcess(true_func=lambda x: np.sin(x) * np.exp(-0.15 * x),
                    mu_0=lambda x: np.sin(x)),

    # One leg missing ancestor
    GaussianProcess(true_func=lambda x: np.where(x < 3, -1, np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x)),

    # Other leg missing ancestor
    GaussianProcess(true_func=lambda x: np.where(x > 6, -1, np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x)),

    # Ancestor on gravel
    GaussianProcess(true_func=lambda x: 0.5 * (np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x)),

    # Ancestor on ice
    GaussianProcess(true_func=lambda x: (np.sin(x) * np.exp(-0.15 * x)) + 0.1 * np.cos(10 * x),
                    mu_0=lambda x: np.sin(x)),

    # Ancestor with true function = simulation function
    GaussianProcess(true_func=lambda x: np.sin(x),
                    mu_0=lambda x: np.sin(x))

]

# ANCESTORS OBSERVE POINTS TO ADAPT
agents[0].observe_true_points([1.6, 7.85])
agents[1].observe_true_points([0.75, 1.6, 2.5, 7.25, 7.6, 7.85, 8.15, 8.5])
agents[2].observe_true_points([1.6, 7.85])
agents[3].observe_true_points([1, 1.6, 2.25, 7.2, 7.85, 8.55])
agents[4].observe_true_points([1.15, 1.6, 7.85])
agents[5].observe_true_points([1.6])

# LAUNCH THE EXPERIMENTS
visualiser = Visualiser()
# visualiser.visualise_ITE_experiment(agents[5], alpha=0.9)
# visualiser.visualise_example_experiment(agents[:])

# TODO: in any collaborative setting with multiple curves being plotted, you need to pass in as options which curves
#  are being plotted. For example, it is unnecssary in the collaborative setting to plot the true function, or the
#  the seen points (this can be seen easily). Maybe you can make the default for some curves not to be seen but you can
#  toggle the option if you so please.

# TODO: think about whether you want to decouple the application parts (all the launch app) from the plotting. I think
#  that they are intriniscally linked because the application is a plottling app and therefore do not believe that they
#  they need different classes.

# TODO: clean up the whole code, and think about whether a better way of displaying the knowledge as the way it is being
#  displayed now. Maybe without all the uncertainties for all the parents... Maybe make that initially hidden and that it
#  can be toggled. That way, you can have all the information at hand, but it is easy to see in the default setting.

# TODO: strech goal but not necessary, make the plotting better whereby you do not re-plot the entire thing each time
#  but only modify certain objects, if you could somehow identify which objects you had to change etc... this is a strech
#  goal as it does not impact the overall project (this is just a toy problem meant for visualisation)

# TODO: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)
