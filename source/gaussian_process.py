import numpy as np


class GaussianProcess:
    def __init__(self,
                 sampling_noise=0,
                 length_scale=1,
                 vertical_scale=0.5,
                 kappa=0.05,
                 kernel="squared_exponential",
                 rho=0.4,
                 alpha=0.9,
                 true_func=lambda x: np.sin(3 * x) + 2 * x,
                 mu_0=lambda x: np.zeros(shape=x.shape[0]),
                 x_seen=np.array([])):
        # GP parameters
        self.sampling_noise = sampling_noise
        self.vertical_scale = vertical_scale
        self.length_scale = length_scale
        self.kappa = kappa
        self.true_func = true_func
        self.mu_0 = mu_0
        self.kernel = kernel
        self.rho = rho

        # Problem parameters
        self.x_start, self.x_stop = 0, 10
        self.num_points = 200
        self.x_problem = np.linspace(start=self.x_start, stop=self.x_stop, num=self.num_points)
        self.x_seen = np.array(x_seen)
        self.y_seen = self.true_func(self.x_seen)
        self.alpha = alpha

    def observe_true_points(self, x):
        x = np.array(x)
        self.x_seen = np.append(self.x_seen, x)
        self.y_seen = np.append(self.y_seen, self.true_func(x))

    def unobserve_true_points(self, x):
        x = np.array(x)
        index = np.where(self.x_seen == x)
        self.x_seen = np.delete(self.x_seen, index)
        self.y_seen = np.delete(self.y_seen, index)

    def query_acquisition_function(self):
        gp_mean = self.mu_new(self.x_problem, self.x_seen, self.y_seen)
        gp_var = self.var_new(self.x_problem, self.x_seen)

        max_val_index = np.argmax(gp_mean + [i * self.kappa for i in gp_var])
        max_val_x_loc = (self.x_stop - self.x_start) * (max_val_index + 1) / self.num_points

        return max_val_x_loc

    def update_gp(self, x_clicked):
        if x_clicked in self.x_seen:
            self.unobserve_true_points(x_clicked)
        else:
            self.observe_true_points(x_clicked)

    def mu_new(self, x, x_seen, y_seen):
        K = self.K_mat(x_seen)
        k = self._k_vec(x, x_seen)
        mu_new = self.mu_0(x) + np.transpose(k) @ np.linalg.inv(
            K + self.sampling_noise * np.identity(n=np.shape(K)[0])) @ (y_seen - self.mu_0(x_seen))
        return mu_new

    def var_new(self, x, x_seen):
        K_computed = self.K_mat(x_seen)
        var = []
        for i in range(len(x)):
            var += [self._kernel_func(x[i], x[i]) + self.sampling_noise - np.transpose(
                self._k_vec(x[i], x_seen)) @ np.linalg.inv(
                K_computed + self.sampling_noise * np.identity(n=np.shape(K_computed)[0])) @ self._k_vec(x[i], x_seen)]
        return var

    def calculate_end_cond_thresh_val(self):
        thresh = self.alpha * np.max(self.mu_new(self.x_problem, self.x_seen, self.y_seen))
        return thresh

    def K_mat(self, x_seen):
        n = len(x_seen)
        mat = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(i, n):
                # Make use of symmetry to halve the computation of the Kernel matrix
                mat[i, j] = self._kernel_func(x_seen[i], x_seen[j])
                mat[j, i] = mat[i, j]
        return mat

    def _kernel_func(self, x1, x2):
        if self.kernel == "squared_exponential":
            return np.exp(-0.5 * ((x2 - x1) / self.length_scale) ** 2)

        elif self.kernel == "matern":
            # Note that the Euclidian distance were given as np.abs(x1-x2): defined for 1-D
            return (1 + ((np.sqrt(5) * np.abs(x1 - x2)) / self.rho) +
                    ((5 * np.abs(x1 - x2) ** 2) / (3 * self.rho ** 2))) * np.exp(
                -((np.sqrt(5) * np.abs(x1 - x2)) / self.rho))

        else:
            print("I'm sorry I do not recognise this kernel. Try: squared_exponential or matern")

    def _k_vec(self, x, x_seen):
        return [self._kernel_func(x, xi) for xi in x_seen]


if __name__ == "__main__":
    gp = GaussianProcess(x_seen=np.array([1, 2, 3]))
    print(gp.x_seen)
    print(gp.y_seen)

    gp2 = GaussianProcess()
    print(gp2.calculate_end_cond_thresh_val())
    gp2.observe_true_points([1.6])
    print(gp2.calculate_end_cond_thresh_val())

# Todo: include an animation component whereby the optimisation can be done automatically wiht a sleep call
#  in between re-plotting.

# Todo: best thing for now is to keep because it does not hurt anything else than readability; meaning, develop
#  the code further to know if it was required or not and then can remove once an MVP of the complete code is done.
