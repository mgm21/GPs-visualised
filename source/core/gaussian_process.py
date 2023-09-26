import numpy as np


class GaussianProcess:
    def __init__(self,
                 true_func=lambda x: np.sin(3 * x) + 2 * x,
                 mu_0=lambda x: np.zeros(shape=x.shape[0]),
                 x_seen=None,
                 x_problem=np.linspace(start=0, stop=10, num=100),
                 sampling_noise=0.001, # Previously 0.1
                 length_scale=1, # Previously 1.5
                 kappa=4,
                 kernel="squared_exponential",
                 rho=0.4,
                 alpha=0.9):

        # GP parameters
        if x_seen is None:
            x_seen = []
        self.sampling_noise = sampling_noise
        self.length_scale = length_scale
        self.kappa = kappa
        self.true_func = true_func
        self.mu_0 = mu_0
        self.kernel = kernel
        self.rho = rho
        self.alpha = alpha

        # Problem parameters
        self.x_problem = x_problem
        self.x_seen = np.array(x_seen)
        self.y_seen = self.true_func(self.x_seen)

    def query_acquisition_function(self):
        gp_mean = self.mu_new(self.x_problem)
        gp_var = self.var_new(self.x_problem)

        max_val_index = np.argmax(gp_mean + [i * self.kappa for i in gp_var])
        max_val_x_loc = (self.x_problem[-1] - self.x_problem[0]) * (max_val_index + 1) / len(self.x_problem)

        return max_val_x_loc

    def mu_new(self, x):
        K = self.K_mat()
        k = self._k_vec(x)
        mu_new = self.mu_0(x) + np.transpose(k) @ np.linalg.inv(
            K + self.sampling_noise * np.identity(n=np.shape(K)[0])) @ (self.y_seen - self.mu_0(self.x_seen))
        return mu_new

    def var_new(self, x):
        K_computed = self.K_mat()
        var = []
        for i in range(len(x)):
            var += [self._kernel_func(x[i], x[i]) + self.sampling_noise - np.transpose(
                self._k_vec(x[i])) @ np.linalg.inv(
                K_computed + self.sampling_noise * np.identity(n=np.shape(K_computed)[0])) @ self._k_vec(x[i])]
        return var

    def K_mat(self):
        n = len(self.x_seen)
        mat = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(i, n):
                # Make use of symmetry to halve the computation of the Kernel matrix
                mat[i, j] = self._kernel_func(self.x_seen[i], self.x_seen[j])
                mat[j, i] = mat[i, j]
        return mat

    def _kernel_func(self, x1, x2):
        if self.kernel == "squared_exponential":
            return np.exp(-0.5 * ((x2 - x1) / self.length_scale) ** 2)

        elif self.kernel == "matern":
            # Note that the Euclidean distance were given as np.abs(x1-x2): defined for 1-D
            return (1 + ((np.sqrt(5) * np.abs(x1 - x2)) / self.rho) +
                    ((5 * np.abs(x1 - x2) ** 2) / (3 * self.rho ** 2))) * np.exp(
                -((np.sqrt(5) * np.abs(x1 - x2)) / self.rho))

        else:
            print("I'm sorry I do not recognise this kernel. Try: squared_exponential or matern")

    def _k_vec(self, x):
        return [self._kernel_func(x, xi) for xi in self.x_seen]

    # TODO: all the methods below could be removed. They are not necessarily the GP's responsibility.
    def observe_true_points(self, x):
        x = np.array(x)
        # In order not to get a singular matrix error, it is important for x_seen to be unique
        self.x_seen = np.unique(np.append(self.x_seen, x))
        self.y_seen = self.true_func(self.x_seen)

    def unobserve_true_points(self, x):
        x = np.array(x)
        index = np.where(self.x_seen == x)
        self.x_seen = np.delete(self.x_seen, index)
        self.y_seen = np.delete(self.y_seen, index)

    def update_seen_point(self, x):
        if x in self.x_seen:
            self.unobserve_true_points(x)
        else:
            self.observe_true_points(x)

    def calculate_end_cond_thresh_val(self):
        thresh = self.alpha * np.max(self.mu_new(self.x_problem))
        return thresh


if __name__ == "__main__":
    gp = GaussianProcess(true_func=np.sin,
                         mu_0=np.cos,
                         x_seen=[],
                         x_problem=np.linspace(start=0, stop=10, num=20))

    gp.observe_true_points(x=gp.query_acquisition_function())
