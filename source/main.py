from gaussian_process import GaussianProcess

# Define the Gaussian Process
agent = GaussianProcess()

# Make agent observe some true points
agent.observe_true_points([2, 5, 6, 7, 8])

# Plot
agent.plot_all(savefig=False)

