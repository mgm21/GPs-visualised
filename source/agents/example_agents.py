from source.core.gaussian_process import GaussianProcess
import numpy as np

agents = [
    # 0 Intact ancestor
    GaussianProcess(true_func=lambda x: np.sin(x) * np.exp(-0.15 * x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # 1 One leg missing ancestor
    GaussianProcess(true_func=lambda x: np.where(x < 3, -1, np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[0.75, 1.6, 2.5, 7.25, 7.6, 7.85, 8.15, 8.5]),

    # 2 Other leg missing ancestor
    GaussianProcess(true_func=lambda x: np.where(x > 6, -1, np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # 3 Ancestor on gravel
    GaussianProcess(true_func=lambda x: 0.5 * (np.sin(x) * np.exp(-0.15 * x)),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1, 1.6, 2.25, 7.2, 7.85, 8.55]),

    # 4 Ancestor on ice
    GaussianProcess(true_func=lambda x: (np.sin(x) * np.exp(-0.15 * x)) + 0.1 * np.cos(10 * x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.15, 1.6, 7.85]),

    # 5 Ancestor with true function = simulation function
    GaussianProcess(true_func=lambda x: np.sin(x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6]),

    # 6 Ancestor whose BD is 90 deg phase shifted (still thinking of a physical reason) compared to the intact one
    GaussianProcess(true_func=lambda x: np.sin(x + np.pi) * np.exp(-0.15 * x),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # 7 Ancestor whose BD is 90 deg phase shifted and does not decay
    GaussianProcess(true_func=lambda x: np.sin(x + np.pi),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # 8 Ancestor whose BD is 90 deg phase shifted and is compressed
    GaussianProcess(true_func=lambda x: 0.5 * np.sin(x + np.pi),
                    mu_0=lambda x: np.sin(x),
                    x_seen=[1.6, 7.85]),

    # 9 Ancestor with a single +ve peak in the middle
    GaussianProcess(true_func=lambda x: np.sin(x*0.5),
                    mu_0=lambda x: np.zeros(x.shape[0]),
                    x_seen=[0, 3, 9]),

    # 10 Ancestor with a single +ve peak at the right
    GaussianProcess(true_func=lambda x: - (np.sin(x*0.5) - 0.5),
                        mu_0=lambda x: np.zeros(x.shape[0]),
                        x_seen=[3, 9]),

    # 11 Ancestor with a single +ve peak in the middle with some compression compared to # 9
    GaussianProcess(true_func=lambda x: (np.sin(x*0.5)) * 0.7,
                            mu_0=lambda x: np.zeros(x.shape[0]),
                            x_seen=[])



]

# Todo: Get the seen points automatically by running ITE on every one of the agents above + on any agent that I want to
#  add in
