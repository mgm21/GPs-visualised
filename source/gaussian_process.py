import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ipywidgets
from dash import Dash, dcc, html, Output, Input, State, callback
import plotly.express as px


class GaussianProcess:
    def __init__(self, sampling_noise=0, length_scale=1, kappa=1):
        # GP parameters
        self.sampling_noise = sampling_noise
        self.length_scale = length_scale
        self.kappa = kappa

        # Problem parameters
        self.x_start, self.x_stop = 0, 10
        self.num_points = 100
        self.x_problem = np.linspace(start=self.x_start, stop=self.x_stop, num=self.num_points)
        self.x_seen = np.array([])
        self.y_seen = np.array([])

    def kernel_func(self, x1, x2):
        # Squared exponential kernel (naive but works for 1-D toy example) see Brochu tutorial p.9
        return np.exp(-0.5 * ((x2 - x1) / self.length_scale) ** 2)

    def mu_0(self, x):
        # TODO: must fix why when mu_0 is different to true_func, the GP behaves in a strange way.
        return np.zeros(shape=x.shape[0])

    # def var_0(self, x):
    #     # Never called
    #     return np.array([self.kernel_func(xi, xi) for xi in x]) + self.sampling_noise

    def true_func(self, x):
        return np.sin(3 * x) + 2 * x

    def k_vec(self, x, x_seen):
        return [self.kernel_func(x, xi) for xi in x_seen]

    def K_mat(self, x_seen):
        n = len(x_seen)
        mat = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(i, n):
                # Make use of symmetry to halve the computation of the Kernel matrix
                mat[i, j] = self.kernel_func(x_seen[i], x_seen[j])
                mat[j, i] = mat[i, j]
        return mat

    def mu_new(self, x, x_seen, y_seen):
        K = self.K_mat(x_seen)
        k = self.k_vec(x, x_seen)
        mu_new = self.mu_0(x) + np.transpose(k) @ np.linalg.inv(
            K + self.sampling_noise * np.identity(n=np.shape(K)[0])) @ (y_seen - self.mu_0(x_seen))
        return mu_new

    def var_new(self, x, x_seen):
        K_computed = self.K_mat(x_seen)
        var = []
        for i in range(len(x)):
            var += [self.kernel_func(x[i], x[i]) + self.sampling_noise - np.transpose(
                self.k_vec(x[i], x_seen)) @ np.linalg.inv(
                K_computed + self.sampling_noise * np.identity(n=np.shape(K_computed)[0])) @ self.k_vec(x[i], x_seen)]
        return var

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
            # # TODO: is this sub-statement necessary? If the optimisation is done it will be stuck anyway.
            # #  removed it for now
            # if x_clicked != self.query_acquisition_function():
            #     # This is a point that the user would like to remove
            #     # If the acquisition is suggesting a point that was already there, do not remove
            self.unobserve_true_points(x_clicked)
        else:
            self.observe_true_points(x_clicked)

        fig = self.generate_plotly_figure()

        return fig

    def generate_plotly_figure(self):
        # Compute/Define the required arrays once
        mu_new = self.mu_new(self.x_problem, self.x_seen, self.y_seen)
        var_new = self.var_new(self.x_problem, self.x_seen)
        xplot = self.x_problem
        y_upper = mu_new + var_new
        y_lower = mu_new - var_new
        # The suggested x location is cast to a numpy array because mu_0 must receive array to behave as intended
        acquisition_x_suggestion = np.array([self.query_acquisition_function()])

        # Build the Plotly figure
        fig = go.FigureWidget([

            # Note: ordering is important for click events (last Scatter added is never covered by others)

            # GP uncertainty
            go.Scatter(x=np.append(xplot, xplot[::-1]),
                       y=np.append(y_upper, y_lower[::-1]),
                       fill='toself',
                       line=dict(color=self.plot_col),
                       name="GP uncertainty"),

            # GP mean
            go.Scatter(x=xplot,
                       y=mu_new,
                       line=dict(color=self.plot_col),
                       name="GP mean"),

            # True function
            go.Scatter(x=xplot,
                       y=self.true_func(xplot),
                       line=dict(color=self.plot_col, dash='dash', width=4),
                       name="True function"),

            # Observed points
            go.Scatter(x=self.x_seen,
                       y=self.true_func(self.x_seen),
                       mode="markers",
                       name="Observed points",
                       marker=dict(color="black", size=10),
                       opacity=0.5),

            # Acquisition function suggestion
            go.Scatter(x=acquisition_x_suggestion,
                       y=self.mu_new(acquisition_x_suggestion, self.x_seen, self.y_seen),
                       marker=dict(color="red", size=10),
                       mode="markers",
                       opacity=0.5,
                       name="Acquisition suggestion")
        ])

        return fig

    def plot_all_matplotlib(self, savefig=True):
        # Set up the plotting environment
        xplot = self.x_problem
        plt.figure()

        # Plot the true, hidden, function
        plt.plot(xplot, self.true_func(xplot), color=self.plot_col, linestyle="--", label="True function", zorder=1)

        # Plot the updated GP mean and uncertainty
        plt.plot(xplot, self.mu_new(xplot, self.x_seen, self.y_seen), color=self.plot_col, label="GP mean", zorder=1)
        # noinspection PyTypeChecker
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

    def start_interactive_gp_dash_app(self):
        # Based on last part of tutorial: https://www.youtube.com/watch?v=pNMWbY0AUJ0&t=1531s
        initial_fig = self.generate_plotly_figure()
        app = Dash(__name__)

        app.layout = html.Div(
            dcc.Graph(figure=initial_fig, id='gp-graph', style={"height": "100vh"})
        )

        @callback(
            Output(component_id='gp-graph', component_property='figure'),
            Input(component_id='gp-graph', component_property='clickData'),
            prevent_initial_call=True
        )
        def update_plot(point_clicked):  # the function argument comes from the component property of the Input
            x = point_clicked['points'][0]['x']

            fig = self.update_gp(x_clicked=x)

            print(point_clicked)
            print(type(point_clicked))
            print(x)

            return fig

        app.run(port=8000)

# TODO:

# TODO: decouple the visualisation from the GP system. Because, you must imagine having many GPs with one visualisation.
#  So, you must have a Plotter/Displayer class which can accept AN ARRAY of GaussianProcess objects.
#  And, it must update them all, whenever someone clicks on the particular curve. Will have to somehow find a way
#  to know which graph has been pressed... I have found the way. When you print the (point_clicked) object in the
#  update_gp method, the meta-data tells you which curve was pressed. If I'm lucky, the curve number will have to do
#  with which GP has been pressed. That said, could also look for the callback documentation, maybe the meta data allows
#  to retrieve the legend of the curve that was pressed. In that case, could start all the legends for one family with
#  the number 1 for example, then we would know that the agent to self.update_gp is the one with that index and the agents
#  could be named accordingly with order that they were passed and therefore plotted.
#  for the plotter class, all the references to 'self' in the above will have to be changed to agent.
#  and some kind of functionality to plot all the agents and only modify the ones that need modifying will have to be
#  included.

# TODO: include an animation component whereby the optimisation can be done automatically wiht a sleep call in between
#  replotting.

# TODO: best thing for now is to keep because it does not hurt anything else than readability; meaning, develop
#  the code further to know if it was required or not and then can remove once an MVP of the complete code is done.

# TODO: think of a way to have a different true_func for every GP, though maybe the same prior...
#  maybe best is to define the true function as both a list (attribute) and a function which produced it
#  and then define functions to move back and forth between the index world and the x value world.

# TODO: ultimately add functionality to remove some points from the "seen_points" this way you don't have to reload
#  the application and you can do most things dynamically.
