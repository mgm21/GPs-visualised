from gaussian_process import GaussianProcess
from visualiser import Visualiser

# Define the Gaussian Process(s)
num_gps = 4
agents = []
for i in range(num_gps):
    agents += [GaussianProcess(kappa=5)]
    agents[i].observe_true_points([3*i])

# Plot
visualiser = Visualiser()
visualiser.plot_gps_matplotlib(agents[:4])


# TODO: try to understand why increasing the sampling noise makes the mean no longer go through the true function
#  curve (is this normal?)

# TODO: try to understand why the trialled optimisation using ITE below with the acquisition did not converge
