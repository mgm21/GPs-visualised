import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from source.core.gaussian_process import GaussianProcess
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats
import matplotlib.font_manager as fm

true_func = lambda x: np.sin(x)
mu_0 = lambda x: np.cos(x)
x_seen = np.array([3.14])
xplot = np.linspace(start=0, stop=10, num=700)
yplot = np.linspace(-4, 6, 700)
z = np.zeros(shape=(len(xplot), len(yplot)))

# GP instantiation
gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

# Probability density values
for i in range(0, len(xplot), 1):
    mu = gp.mu_new(xplot)[i]
    std = np.sqrt(gp.var_new(xplot)[i])
    z[xplot==xplot[i], :] = stats.norm.pdf(yplot, mu, std)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xmesh, ymesh = np.meshgrid(xplot, yplot)
surf = ax.plot_surface(xmesh, ymesh, z.T, cmap=cm.viridis,
                       linewidth=0, antialiased=False, vmin=0, vmax=1)
ax.set_zlim(0, 1.)
fig.colorbar(surf, shrink=0.3, aspect=10)

# To edit the font and labels of the axes
ax.view_init(45, 45) # Change to ax.view_init(25, 45) to recover 3d view
font_used = "Charter"
font_size = 16
font = {'fontname': font_used}
ax.set_xlabel("x", fontsize=font_size, **font)
ax.set_ylabel("f", fontsize=font_size, **font)
ax.set_zlabel("p", fontsize=font_size, **font)

# To scale the axes' size
x_scale=2
y_scale=1.5
z_scale=1
scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)
ax.get_proj=short_proj

# To remove the axes numbers from the p (z) axis
ax = plt.gca()
ax.zaxis.set_ticklabels([])
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)


fig.savefig("3d_contour.png", dpi=600,)

plt.show()