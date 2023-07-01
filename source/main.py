from gaussian_process import GaussianProcess

# Define the Gaussian Process
# TODO: try to understand why increasing the sampling noise makes the mean no longer go through the true function curve (is this normal?)
agent = GaussianProcess(sampling_noise=0, kappa=5)

# TODO: try to understand why the trialled optimisation using ITE below with the acquisition did not converge
# Make agent observe some true points
agent.observe_true_points([5, 7])
print(agent.query_acquisition_function())

# Plot
agent.plot_all(savefig=False)

