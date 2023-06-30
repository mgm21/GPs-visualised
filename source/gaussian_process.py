import numpy as np
import matplotlib.pyplot as plt


class GaussianProcess:
    def __init__(self, sampling_noise=0.01, amplitude=1, length_scale=1):
        # GP parameters
        self.sampling_noise = sampling_noise
        self.amplitude = amplitude
        self.length_scale = length_scale

        # Problem parameters
        self.x_start, self.x_stop = 0, 10
        self.num_points = 100
        self. x_problem = np.linspace(start=self.x_start, stop=self.x_stop, num=self.num_points)
        self.x_seen = []
        self.y_seen = []

        # Plotting parameters
        self.plot_col = "cornflowerblue"

    def k(self, x1, x2):
        return self.amplitude ** 2 * np.exp(-0.5 * (np.abs(x1 - x2) / self.length_scale) ** 2)

    def mu_0(self, x):
        return np.array([0 for _ in x])

    def var_0(self, x):
        return np.array([self.k(xi, xi) for xi in x]) + self.sampling_noise

    # TODO: think of a way to have a different true_func for every GP, though maybe the same prior...
    def true_func(self, x):
        return np.sin(x)

    # TODO: replace all the x_seen, y_seen, etc... occurrences with the attributes from the class self.x_seen,
    #  etc... (maybe?)
    def k_vec(self, x, x_seen):
        return [self.k(x, xi) for xi in x_seen]

    def K_mat(self, x_seen):
        mat = np.zeros(shape=(len(x_seen), len(x_seen)))
        for i in range(len(x_seen)):
            for j in range(len(x_seen)):
                mat[i, j] = self.k(x_seen[i], x_seen[j])
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
            var += [self.k(x[i], x[i]) + self.sampling_noise - np.transpose(self.k_vec(x[i], x_seen)) @ np.linalg.inv(
                K_computed + self.sampling_noise * np.identity(n=np.shape(K_computed)[0])) @ self.k_vec(x[i], x_seen)]

        return var

    def observe_true_points(self, x):
        self.x_seen.extend(x)
        self.y_seen.extend(self.true_func(x))

    def query_acquisition_function(self):
        # Returns
        return (self.x_stop - self.x_start) \
            * (np.argmax(self.mu_new(self.x_problem, self.x_seen, self.y_seen)) + 1) \
            / self.num_points


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

        # Edit plot layout
        # plt.tick_params(left=False, right=False, labelleft=False,
        #                 labelbottom=False, bottom=False)
        plt.legend()

        if savefig:
            plt.savefig("example-sin", dpi=50)

        plt.show()
