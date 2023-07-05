import numpy as np


class ITE:
    def __init__(self, gp, max_num_steps=20):
        self.gp = gp
        self.max_num_steps = max_num_steps
        self.counter = 0

    def perform_full_adaptation(self):
        while not self.adaptation_is_done():
            self.take_adaptation_step()
            self.counter += 1
        self.counter = 0

    def take_adaptation_step(self):
        acquisition_x_suggestion = self.gp.query_acquisition_function()
        self.gp.observe_true_points(acquisition_x_suggestion)

    def adaptation_is_done(self):
        alpha = 0.9  # temporary line before I figure out where to put the threshold calculation
        end_cond_thresh = alpha * np.max(self.gp.mu_new(self.gp.x_problem))
        return self.counter >= self.max_num_steps or np.any(self.gp.y_seen > end_cond_thresh)


if __name__ == "__main__":
    from source.agents.example_agents import agents

    index_agent_to_optimise = 1

    ite = ITE(agents[index_agent_to_optimise])
    ite.perform_full_adaptation()
    print(ite.gp.x_seen)

    max_fitness_found = max(ite.gp.y_seen)
    bd_of_max_fitness = ite.gp.x_seen[ite.gp.y_seen == max_fitness_found]

    print(f"max fitness found: {max_fitness_found}")
    print(f"bd of max fitness: {bd_of_max_fitness}")
