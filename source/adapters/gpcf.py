import numpy as np
from scipy import optimize


class GPCF:
    def __init__(self, past_gp_arr, current_gp, max_num_steps=20):
        self.past_gp_arr = np.array(past_gp_arr)
        self.current_gp = current_gp
        self.max_num_steps = max_num_steps
        self.counter = 0
        self.W = self._initialise_weights()

    def perform_full_adaptation(self):
        while not self.adaptation_is_done():
            self.take_adaptation_step()
            self.counter += 1
        self.counter = 0

    def take_adaptation_step(self):
        acquisition_x_suggestion = self.current_gp.query_acquisition_function()
        self.current_gp.observe_true_points(acquisition_x_suggestion)
        self.update_current_gp_mu_0()

    def update_current_gp_mu_0(self):
        self._optimise_weights()
        self.current_gp.mu_0 = lambda x: self._get_past_gp_mu_vals(x).T @ self.W

    def _initialise_weights(self):
        num_past_gps = self.past_gp_arr.shape[0]
        return np.repeat(a=1 / num_past_gps, repeats=num_past_gps)

    def _optimise_weights(self):
        optim_result = optimize.minimize(fun=self._get_negative_loglikelihood, x0=self.W)
        self.W = optim_result.x
        # Normalise weights
        # Todo: should I be normalising the weights? Is this how it should be done? in [0, 1] or [-1, 1]?
        self.W = self.W / sum(np.abs(self.W))
        print(f"weights: {self.W}")

    def _get_past_gp_mu_vals(self, x):
        # There are as many mu_vals as there are past/ancestor gaussian processes
        mu_vals = np.array([gp.mu_new(x) for gp in self.past_gp_arr])
        return mu_vals

    def _get_negative_loglikelihood(self, W):
        # TODO: check the base of the log in Rasmussen
        # TODO: Is it acceptable that this function crashes when current GP has not seen any points?
        A = self.current_gp.y_seen - self._get_past_gp_mu_vals(self.current_gp.x_seen).T @ W
        K = self.current_gp.K_mat()
        t = len(self.current_gp.y_seen)
        neg_llh = -(-0.5 * A.T @ np.linalg.inv(K) @ A - 0.5 * np.log(np.linalg.det(K)) - (t / 2) * np.log(2 * np.pi))
        # print(f"This is the neg llh: {neg_llh}")
        return neg_llh

    def adaptation_is_done(self):
        alpha = 0.9  # temporary line before I figure out where to put the threshold calculation
        end_cond_thresh = alpha * np.max(self.current_gp.mu_new(self.current_gp.x_problem))
        return self.counter >= self.max_num_steps or np.any(self.current_gp.y_seen > end_cond_thresh)


if __name__ == "__main__":
    from source.agents.example_agents import agents

    agents[0].observe_true_points([1.6, 7.85])
    # agents[1].observe_true_points([0.75, 1.6, 2.5, 7.25, 7.6, 7.85, 8.15, 8.5])
    # agents[2].observe_true_points([1.6, 7.85])
    # agents[3].observe_true_points([1, 1.6, 2.25, 7.2, 7.85, 8.55])
    # agents[4].observe_true_points([1.15, 1.6, 7.85])
    # agents[5].observe_true_points([1.6])

    i = 1

    current = agents[i]
    ancestors = agents[0:i] + agents[i + 1:]

    print(current)

    # Unobserve all the observed points of the current robot
    current.unobserve_true_points([current.x_seen])

    gpcf_adapter = GPCF(ancestors, current)

    gpcf_adapter.perform_full_adaptation()
    print(gpcf_adapter.current_gp.x_seen)
