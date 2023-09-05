from source.core.gaussian_process import GaussianProcess
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

true_func = lambda x: np.sin(x)
mu_0 = lambda x: np.cos(x)
x_seen = [3.14]
xplot = np.linspace(start=0, stop=10, num=200)
yplot = np.linspace(-5, 5, 200)

# GP instantiation
gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

# Plot config
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set(
       xlabel='Behavioural descriptor',
       ylabel='Fitness',
       zlabel='Probability')

# Probability density plotting
n = 3
for i in range(0, len(xplot), n):
    if i % 2 == 0:
        color = "purple"
    else:
        color = "cornflowerblue"
    mu = gp.mu_new(xplot)[i]
    std = np.sqrt(gp.var_new(xplot)[i])
    x = np.repeat(a=xplot[i], repeats=len(yplot))
    ax.plot3D(x, yplot, stats.norm.pdf(yplot, mu, std), color=color)

ax.view_init(25, 45)
fig.savefig("3d_gp_matplotlib.png", dpi=400)
plt.show()


