from gaussian_process import GaussianProcess

# Define the Gaussian Process
agent = GaussianProcess()

# Make agent observe some true points
agent.observe_true_points([])
print(agent.query_acquisition_function())

# Plot
agent.plot_all(savefig=False)

