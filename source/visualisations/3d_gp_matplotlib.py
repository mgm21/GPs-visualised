from source.core.gaussian_process import GaussianProcess
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

true_func = lambda x: np.sin(x)
mu_0 = lambda x: np.cos(x)
x_seen = [3.14]
xplot = np.linspace(start=0, stop=10, num=500)
yplot = np.linspace(-5, 5, 500)

# GP instantiation
gp = GaussianProcess(true_func=true_func, mu_0=mu_0, x_seen=x_seen, x_problem=xplot)

# Plot config
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set(
       xlabel='x',
       ylabel='f',
       zlabel='p')

# Probability density plotting
n = 21 # 151 (4), 25 (20), 5 (100)
print((len(xplot)+1)/n)
counter = 0
for i in range(0, len(xplot), n):
    counter += 1
    if i % 2 == 0:
        color = "purple"
    else:
        color = "cornflowerblue"
    mu = gp.mu_new(xplot)[i]
    std = np.sqrt(gp.var_new(xplot)[i])
    x = np.repeat(a=xplot[i], repeats=len(yplot))
    ax.plot3D(x, yplot, stats.norm.pdf(yplot, mu, std), color=color, alpha=0.7)

print(counter)

# ax.view_init(25, 45) # Angled view
ax.view_init(45, 150)
font_used = "Charter"
font_size = 16
font = {'fontname': font_used}
ax.set_xlabel("x", fontsize=font_size, **font)
ax.set_ylabel("f", fontsize=font_size, **font)
ax.set_zlabel("p", fontsize=font_size, **font)
fig.savefig("3d_gp_formation.png", dpi=600)

plt.show()


