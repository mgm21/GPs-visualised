from gaussian_process import GaussianProcess
from visualiser import Visualiser
import numpy as np

# Define the Gaussian Process(s)
agents = [
    GaussianProcess(true_func=lambda x: np.sin(3.5 * x) + 2 * x,
                    mu_0=lambda x: np.sin(3.5 * x) + 2.5 * x),

    GaussianProcess(true_func=lambda x: np.sin(2 * x) + 2 * x,
                    mu_0=lambda x: np.zeros(x.shape[0]))
]

# Plot
visualiser = Visualiser()
visualiser.start_interactive_gp_dash_app(agents)
# visualiser.generate_plotly_figure(agents[:4])
# visualiser.plot_gps_matplotlib(agents[:4])


# TODO: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)

# TODO: try to understand why the trialled optimisation using ITE below with the acquisition did not converge
