from source.core.gaussian_process import GaussianProcess
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

true_func = lambda x: np.sin(x)
mu_0 = lambda x: np.cos(x)
x_seen = np.array([3.14])
xplot = np.linspace(start=0, stop=10, num=200)
yplot = np.linspace(-5, 5, 300)
z = np.zeros(shape=(len(xplot), len(yplot)))

# GP instantiation
gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

# Probability density values
for i in range(0, len(xplot), 51):
    mu = gp.mu_new(xplot)[i]
    std = np.sqrt(gp.var_new(xplot)[i])
    z[xplot==xplot[i], :] = stats.norm.pdf(yplot, mu, std)

# Plot config
fig = go.Figure(data=[go.Surface(z=z.T, x=xplot, y=yplot)],)
trace=dict(type='surface',
           colorbar=dict(lenmode='fraction', len=0.1, thickness=20))
fig.add_traces(trace)
fig.update_layout(
    autosize=True,
    scene=dict(
        xaxis_title='x',
        yaxis_title='f',
        zaxis_title='p',
    ),
)

fig.show()