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

# TODO: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)
