import numpy as np
from scipy import optimize


class GPCF:
    def __init__(self, past_gp_arr, current_gp):
        self.W = None
        # TODO: check when you should cast objects to arrays (maybe this is already being passed as an array?)
        self.past_gp_arr = np.array(past_gp_arr)
        self.current_gp = current_gp

    def perform_full_adaptation(self):
        ...

    def take_adaptation_step(self):
        ...

    def update_current_gp_mu_0(self):
        self.optimise_weights()
        self.current_gp.mu_0 = lambda x: self.get_past_gp_mu_vals(x) @ self.W

    def optimise_weights(self):
        if len(self.current_gp.x_seen) == 0:
            num_past_gps = self.past_gp_arr.shape[0]
            self.W = np.repeat(a=1/num_past_gps, repeats=num_past_gps)
        else:
            ...


    def get_past_gp_mu_vals(self, x):
        # There are as many mu_vals as there are past/ancestor gaussian processes
        mu_vals = np.array([gp.mu_new(x, gp.x_seen, gp.y_seen) for gp in self.past_gp_arr])
        return mu_vals

    def loglikelihood(self, W):
        # TODO: check the basis of the log in Rasmussen
        # TODO: check all the types here (arrays, shapes for matrix multiplications, etc...)
        A = self.current_gp.y_seen - self.get_past_gp_mu_vals(self.current_gp.x_seen) @ W
        K = self.current_gp.K_mat
        t = len(self.current_gp.y_seen)
        return -0.5 * A.T @ np.linalg.inv(K) @ A - 0.5 * np.log(np.abs(K)) - (t/2) * np.log(2*np.pi)


if __name__ == "__main__":
    from source.agents.example_agents import agents

    agents[0].observe_true_points([1.6, 7.85])
    agents[1].observe_true_points([0.75, 1.6, 2.5, 7.25, 7.6, 7.85, 8.15, 8.5])

    ancestors = agents[0:2]
    current = agents[2]

    gpcf_adapter = GPCF(ancestors, current)
    print(gpcf_adapter.get_past_gp_mu_vals(1.6))
    print(gpcf_adapter.current_gp.x_seen)
    print(gpcf_adapter.W)
    gpcf_adapter.optimise_weights()
    print(gpcf_adapter.W)


# TODO: need to change all the x_seen getting passed to methods. Really, should not be too long.
#  once you change the methods itself in gaussian_process, the job simply becomes to remove the
#  arguments from all the other class calls of that method
