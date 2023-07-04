from source.gaussian_process import GaussianProcess
import numpy as np

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