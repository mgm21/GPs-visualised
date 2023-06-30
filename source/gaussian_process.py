import numpy as np
import matplotlib.pyplot as plt


class GaussianProcess:
    def __init__(self, sampling_noise=0.01, length_scale=1, kappa=1):
        # GP parameters
        self.sampling_noise = sampling_noise
        self.length_scale = length_scale
        self.kappa = kappa

        # Problem parameters (akin to quality-diversity behavioural descriptor bounds and resolution)
        self.x_start, self.x_stop = 0, 10
        self.num_points = 100
        self.x_problem = np.linspace(start=self.x_start, stop=self.x_stop, num=self.num_points)
        self.x_seen = []
        self.y_seen = []

        # Plotting parameters
        self.plot_col = "cornflowerblue"
        self.show_axes_ticks_labels = True

    def kernel_func(self, x1, x2):
        # Squared exponential kernel (naive but works for 1-D toy example) see Brochu tutorial p.9
        return np.exp(-0.5 * (np.abs(x2 - x1) / self.length_scale) ** 2)

    def mu_0(self, x):
        return np.zeros(shape=len(x))

    def var_0(self, x):
        # Never called.
        return np.array([self.kernel_func(xi, xi) for xi in x]) + self.sampling_noise

    # TODO: think of a way to have a different true_func for every GP, though maybe the same prior...
    #  maybe best is to define the true function as both a list (attribute) and a function which produced it
    #  and then define functions to move back and forth between the index world and the x value world.
    #  BOOKMARK
    def true_func(self, x):
        return [sum(x) for x in zip([np.sin(3*j) for j in x], [2 * i for i in x])]

    # TODO: replace all the x_seen, y_seen, etc... occurrences with the attributes from the class self.x_seen,
    #  etc... (maybe?): think about how it will work when you have several
    # TODO: best thing for now is to keep because it does not hurt anything else than readability; meaning, develop
    #  the code further to know if it was required or not and then can remove once an MVP of the complete code is done.
    def k_vec(self, x, x_seen):
        return [self.kernel_func(x, xi) for xi in x_seen]

    def K_mat(self, x_seen):
        mat = np.zeros(shape=(len(x_seen), len(x_seen)))
        for i in range(len(x_seen)):
            for j in range(len(x_seen)):
                mat[i, j] = self.kernel_func(x_seen[i], x_seen[j])
        return mat

    def mu_new(self, x, x_seen, y_seen):
        K_computed = self.K_mat(x_seen)
        k_computed = self.k_vec(x, x_seen)
        return self.mu_0(x) + np.transpose(k_computed) @ np.linalg.inv(
            K_computed + self.sampling_noise * np.identity(n=np.shape(K_computed)[0])) @ (y_seen - self.mu_0(x_seen))

    def var_new(self, x, x_seen):
        K_computed = self.K_mat(x_seen)

        var = []
        for i in range(len(x)):
            var += [self.kernel_func(x[i], x[i]) + self.sampling_noise - np.transpose(
                self.k_vec(x[i], x_seen)) @ np.linalg.inv(
                K_computed + self.sampling_noise * np.identity(n=np.shape(K_computed)[0])) @ self.k_vec(x[i], x_seen)]

        return var

    def observe_true_points(self, x):
        self.x_seen.extend(x)
        self.y_seen.extend(self.true_func(x))

    def query_acquisition_function(self):

        gp_mean = self.mu_new(self.x_problem, self.x_seen, self.y_seen)
        gp_var = self.var_new(self.x_problem, self.x_seen)

        max_val_index = np.argmax(gp_mean + [i * self.kappa for i in gp_var])
        max_val_xloc = (self.x_stop - self.x_start) * (max_val_index + 1) / self.num_points

        return max_val_xloc

    def plot_all(self, savefig=True):
        # Set up the plotting environment
        xplot = self.x_problem
        plt.figure()

        # Plot the true, hidden, function
        plt.plot(xplot, self.true_func(xplot), color=self.plot_col, linestyle="--", label="True function", zorder=1)

        # Plot the updated GP mean and uncertainty
        plt.plot(xplot, self.mu_new(xplot, self.x_seen, self.y_seen), color=self.plot_col, label="GP mean", zorder=1)
        plt.fill_between(xplot,
                         self.mu_new(xplot, self.x_seen, self.y_seen) - self.var_new(xplot, self.x_seen),
                         self.mu_new(xplot, self.x_seen, self.y_seen) + self.var_new(xplot, self.x_seen),
                         color=self.plot_col,
                         alpha=0.4)

        # Plot the "seen" points
        plt.scatter(self.x_seen, self.true_func(self.x_seen), color="black", marker=".", label="Observed points",
                    zorder=2, linewidths=1, s=40,
                    alpha=0.4)

        plt.legend()

        # Edit plot layout
        if self.show_axes_ticks_labels:
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)

        if savefig:
            plt.savefig("example-sin", dpi=20)

        plt.show()
