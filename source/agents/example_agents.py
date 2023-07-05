from source.core.gaussian_process import GaussianProcess
import numpy as np

agents = [
    # Intact ancestor
    GaussianProcess(true_func=lambda x: np.sin(x) * np.exp(-0.15 * x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # One leg missing ancestor
    GaussianProcess(true_func=lambda x: np.where(x < 3, -1, np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[0.75, 1.6, 2.5, 7.25, 7.6, 7.85, 8.15, 8.5]),

    # Other leg missing ancestor
    GaussianProcess(true_func=lambda x: np.where(x > 6, -1, np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # Ancestor on gravel
    GaussianProcess(true_func=lambda x: 0.5 * (np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1, 1.6, 2.25, 7.2, 7.85, 8.55]),

    # Ancestor on ice
    GaussianProcess(true_func=lambda x: (np.sin(x) * np.exp(-0.15 * x)) + 0.1 * np.cos(10 * x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.15, 1.6, 7.85]),

    # Ancestor with true function = simulation function
    GaussianProcess(true_func=lambda x: np.sin(x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6])

]

# Todo: Get the seen points automatically by running ITE on every one of the agents above + on any agent that I want to
#  add in
