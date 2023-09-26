import plotly.graph_objects as go
import numpy as np

from source.core.gaussian_process import GaussianProcess
import scipy.stats as stats
import numpy as np
import kaleido
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly

import plotly.express as px


true_func = lambda x: np.sin(x)
mu_0 = lambda x: np.cos(x)
x_seen = [3.14]
xplot = np.linspace(start=0, stop=10, num=250)
yplot = np.linspace(-3, 3, 500)

# GP instantiation
gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

# Plot config
fig = px.scatter_3d()

# Probability density plotting
counter = 0 # 151 (4), 25 (20), 5 (100)

for n in [91, 71, 51, 31, 21, 11, 5, 3, 1]:
    counter += 1
    if counter < 10:
        counter_str = f"00{counter}"
    elif counter < 100:
        counter_str = f"0{counter}"
    else:
        counter_str = f"{counter}"

# TODO: in the below, switch x and y in the parameter names (not the passed args) to recover original behaviour

    for i in range(0, len(xplot), n):
        mu = gp.mu_new(xplot)[i]
        std = np.sqrt(gp.var_new(xplot)[i])
        x = np.repeat(a=xplot[i], repeats=len(yplot))
        prob = stats.norm.pdf(yplot, mu, std)
        # ax.scatter(x, yplot, prob, c=prob, cmap='viridis', vmin=0, vmax=1, alpha=1,)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=yplot,
                z=prob,
                showlegend=False,
                mode="markers",
                marker=dict(
                    colorbar=dict(thickness=10, len=0.4),
                    size=2,
                    color=prob,
                    colorscale='Viridis',
                    opacity=1,
                    cmin=0,
                    cmax=0.8,
                ),
            ),
        )

    camera = dict(
        # up=dict(x=0, y=0, z=0),
        # center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=2, z=2)
    )

    fig.update_layout(scene_camera=camera,
                      scene = dict(
                        xaxis_title='x',
                        yaxis_title='y',
                        zaxis_title='Probability',
                        aspectmode="manual",
                        aspectratio=dict(x=1.5, y=1, z=1),
                        xaxis=dict(range=[0, 10], showticklabels=False),
                        yaxis=dict(range=[-4, 4], showticklabels=False),
                        zaxis=dict(range=[0, 1], showticklabels=False)),

                        width=600,
                        height=600,
                        margin=dict(r=0, b=0, l=0, t=0),)


    fig.write_image(f"images/counter_{counter_str}.png", scale=5)