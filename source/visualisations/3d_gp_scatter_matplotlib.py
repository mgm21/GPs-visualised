from source.core.gaussian_process import GaussianProcess
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


true_func = lambda x: np.sin(x)
mu_0 = lambda x: np.cos(x)
x_seen = [3.14]
xplot = np.linspace(start=0, stop=10, num=250)
yplot = np.linspace(-3, 3, 500)

# GP instantiation
gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

# Plot config
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set(xlabel='x', ylabel='f', zlabel='p')

# Probability density plotting
n = 1 # 151 (4), 25 (20), 5 (100)
for i in range(0, len(xplot), n):
    mu = gp.mu_new(xplot)[i]
    std = np.sqrt(gp.var_new(xplot)[i])
    x = np.repeat(a=xplot[i], repeats=len(yplot))
    prob = stats.norm.pdf(yplot, mu, std)
    ax.scatter(x, yplot, prob, c=prob, cmap='viridis', vmin=0, vmax=1, alpha=1,)

ax.view_init(45, 45)
# font_used = "Charter"
# font_size = 16
# font = {'fontname': font_used}
# ax.set_xlabel("x", fontsize=font_size, **font)
# ax.set_ylabel("f", fontsize=font_size, **font)
# ax.set_zlabel("p", fontsize=font_size, **font)
# fig.savefig("3d_gp_scatter.png", dpi=300)
plt.show()


