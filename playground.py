import numpy as np
import matplotlib.pyplot as plt

def k(x1, x2, s=1, l=1):
    return s**2*np.exp(-0.5*(np.abs(x1-x2)/l)**2)

def mu_0(x):
    return np.array([0 for _ in x])

def var_0(x):
    return np.array([k(xi, xi) for xi in x]) + noise

def true_func(x):
    return np.sin(x)

def k_vec(x, x_seen):
    return [k(x, xi) for xi in x_seen]

def K_mat(x_seen):
    mat = np.zeros(shape=(len(x_seen), len(x_seen)))
    for i in range(len(x_seen)):
        for j in range(len(x_seen)):
            mat[i, j] = k(x_seen[i], x_seen[j])
    return mat

def mu_new(x, x_seen, y_seen):
    K_computed = K_mat(x_seen)
    k_computed = k_vec(x, x_seen)
    return mu_0(x) + np.transpose(k_computed) @ np.linalg.inv(K_computed + noise * np.identity(n=np.shape(K_computed)[0])) @ (y_seen-mu_0(x_seen))

def var_new(x, x_seen):
    K_computed = K_mat(x_seen)

    var = []
    for i in range(len(x)):
        var += [k(x[i], x[i]) + noise - np.transpose(k_vec(x[i], x_seen)) @ np.linalg.inv(K_computed + noise * np.identity(n=np.shape(K_computed)[0])) @ k_vec(x[i], x_seen)]

    return var

# Visualise the prior (can also just set x_seen = [] to get the same behaviour below)
# plt.figure()
# plt.plot(xplot, mu_0(xplot))
# plt.fill_between(xplot, mu_0(xplot) - var_0(xplot), mu_0(xplot) + var_0(xplot), alpha=0.4)
# plt.xlim(0, 10)
# plt.ylim(-2, 2)
# plt.show()

# Parameters
noise = 0.015
x_start, x_stop = 0, 10
x_seen = [1, 3, 4, 6, 7, 9]
y_seen = true_func(x_seen)
xplot = np.linspace(start=x_start, stop=x_stop, num=100)

# Plot
plt.figure()

# Visualise the true function
func_colour = "cornflowerblue"
plt.plot(xplot, true_func(xplot), color=func_colour, linestyle="--", label="True function", zorder=1)

# Visualise the updated GP
gp_colour = "cornflowerblue"

plt.xlim(x_start, x_stop)
plt.ylim(-2, 2)

plt.plot(xplot, mu_new(xplot, x_seen, y_seen), color=gp_colour, label="GP mean", zorder=1)
plt.fill_between(xplot,
                 mu_new(xplot, x_seen, y_seen) - var_new(xplot, x_seen),
                 mu_new(xplot, x_seen, y_seen) + var_new(xplot, x_seen),
                 color=gp_colour,
                 alpha=0.4)

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

# Visualise points common to true function and GP "seen points"
plt.scatter(x_seen, true_func(x_seen), color="black", marker=".", label="Observed points", zorder=2, linewidths=1, s=40, alpha=0.4)


plt.legend()
plt.savefig("example-sin", dpi=50)
plt.show()

# Note for notion:
# - [ok] Now you can start putting your GPs directly into your reports, and built by you.
# - [done] Add all the sigma_noise where they are missing, I did not put them in
# - [done] Add mu_0 so that in the absence of seen points you can still view the GP
# - [MUST ASK ANTOINE] Could not do the var_new all in a vector way because think about the shapes if you do the equation with
#   k vector (x by x_seen) it will end up with the wrong shapes. Bring this up with Antoine.
# - [tried for now, want to get mvp] try to remove for loops
# - [done] clean the code
# - make it so that you can dynamically update the graph (find out how to make an interactive graph
# in a new library so that you can just click on the plot where you would like to add a new point: find
# a library that can do this!)
# - clean everything up and implement with jax (even if jax won't be very useful here)


