from gaussian_process import GaussianProcess
from visualiser import Visualiser
import numpy as np

# Define the Gaussian Process(s)
agents = [
    GaussianProcess(true_func=lambda x: np.sin(3.5 * x) + 2 * x,
                    mu_0=lambda x: np.sin(3.5 * x) + 2.5 * x,
                    kappa=0.05,
                    rho=0.4,
                    kernel="matern"),

    GaussianProcess(true_func=lambda x: np.sin(2 * x) + 2 * x,
                    mu_0=lambda x: np.zeros(x.shape[0])),

    GaussianProcess(true_func=lambda x: np.sin(4 * x) + 3 * x,
                    mu_0=lambda x: np.zeros(x.shape[0]))
]

# Experiments
visualiser = Visualiser()
visualiser.visualise_ITE_experiment(agents[0], alpha=0.9)


# TODO: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)

# TODO: try to understand why the trialled optimisation using ITE below with the acquisition did not converge
