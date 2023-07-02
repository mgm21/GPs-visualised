import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ipywidgets
from dash import Dash, dcc, html, Output, Input, State, callback
import plotly.express as px


class Visualiser:
    def __init__(self):
        self.plot_cols = ["cornflowerblue", "tan", "seagreen", "mediumorchid", "silver", "salmon", ]

        # Plotly-specific
        self.num_plotly_objects_per_gp = 5

        # Matplotlib-specific
        self.show_axes_ticks_labels = True

    def generate_plotly_figure(self, gps_arr):

        # Initialise the overall Plotly figure
        fig = go.FigureWidget()

        for i, gp in enumerate(gps_arr):
            # Compute/Define the required arrays once
            gp.plot_col = self.plot_cols[i]
            mu_new = gp.mu_new(gp.x_problem, gp.x_seen, gp.y_seen)
            var_new = gp.var_new(gp.x_problem, gp.x_seen)
            xplot = gp.x_problem
            y_upper = mu_new + var_new
            y_lower = mu_new - var_new
            # The suggested x location is cast to a numpy array because mu_0 must receive array to behave as intended
            acquisition_x_suggestion = np.array([gp.query_acquisition_function()])

            # Note: ordering is important for click events (last Scatter added is never covered by others)
            # Note: make sure that the number of objects below matches self.num_plotly_objects_per_gp attribute

            # GP uncertainty
            fig.add_scatter(x=np.append(xplot, xplot[::-1]),
                            y=np.append(y_upper, y_lower[::-1]),
                            fill='toself',
                            line=dict(color=gp.plot_col),
                            name=f"{i} GP uncertainty"),

            # GP mean
            fig.add_scatter(x=xplot,
                            y=mu_new,
                            line=dict(color=gp.plot_col),
                            name=f"{i} GP mean"),

            # True function
            fig.add_scatter(x=xplot,
                            y=gp.true_func(xplot),
                            line=dict(color=gp.plot_col, dash='dash', width=4),
                            name=f"{i} True function"),

            # Observed points
            fig.add_scatter(x=gp.x_seen,
                            y=gp.true_func(gp.x_seen),
                            mode="markers",
                            name=f"{i} Observed points",
                            marker=dict(color=gp.plot_col, size=10),
                            opacity=0.5),

            # Acquisition function suggestion
            fig.add_scatter(x=acquisition_x_suggestion,
                            y=gp.mu_new(acquisition_x_suggestion, gp.x_seen, gp.y_seen),
                            marker=dict(color="black", size=10, symbol="cross"),
                            mode="markers",
                            opacity=0.5,
                            name=f"{i} Acquisition suggestion")

        return fig

    def visualise_example_experiment(self, gps_arr):
        # Based on last part of tutorial: https://www.youtube.com/watch?v=pNMWbY0AUJ0&t=1531s
        initial_fig = self.generate_plotly_figure(gps_arr)

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

            index_clicked_curve = point_clicked['points'][0]['curveNumber']
            gp_index = index_clicked_curve // self.num_plotly_objects_per_gp
            print(index_clicked_curve)
            print(gp_index)

            gps_arr[gp_index].update_gp(x_clicked=x)
            fig = self.generate_plotly_figure(gps_arr)

            print(point_clicked)
            print(type(point_clicked))
            print(x)

            return fig

        app.run(port=8000)

    def add_ITE_threshold_to_fig(self, fig, gp, alpha):
        # TODO: move the threshold_val calculation to the GaussianProcess class.
        threshold_val = np.repeat(a=alpha * np.max(gp.mu_new(gp.x_problem, gp.x_seen, gp.y_seen)),
                                  repeats=len(gp.x_problem))
        fig.add_scatter(x=gp.x_problem,
                        y=threshold_val,
                        line=dict(color="red", dash='dash', width=4),
                        name=f"End condition")

        return fig

    def visualise_ITE_experiment(self, gp, alpha):
        initial_fig = self.generate_plotly_figure([gp])
        # TODO: make the following into a function to not repeat the code below

        initial_fig = self.add_ITE_threshold_to_fig(initial_fig, gp, alpha)
        initial_fig.update_layout(
            title=dict(text="ITE Experiment", font=dict(size=50), automargin=False, yref='paper'),
            title_x=0.5
        )

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

            index_clicked_curve = point_clicked['points'][0]['curveNumber']
            gp_index = index_clicked_curve // self.num_plotly_objects_per_gp

            # print(index_clicked_curve)
            # print(gp_index)

            gp.update_gp(x_clicked=x)
            fig = self.generate_plotly_figure([gp])
            fig = self.add_ITE_threshold_to_fig(fig, gp, alpha)
            fig.update_layout(
                title=dict(text="ITE Experiment", font=dict(size=50), automargin=False, yref='paper'),
                title_x=0.5
            )

            # print(point_clicked)
            # print(type(point_clicked))
            # print(x)

            return fig

        app.run(port=8000)

    def plot_gps_matplotlib(self, gps_arr, savefig=True):
        # Set up the plotting environment
        plt.figure()

        for i, gp in enumerate(gps_arr):

            gp.plot_col = self.plot_cols[i]

            xplot = gp.x_problem

            # Plot the true, hidden, function
            plt.plot(xplot, gp.true_func(xplot), color=gp.plot_col, linestyle="--", label=f"{i}: True function",
                     zorder=1)

            # Plot the updated GP mean and uncertainty
            plt.plot(xplot, gp.mu_new(xplot, gp.x_seen, gp.y_seen), color=gp.plot_col, label=f"{i}: GP mean", zorder=1)
            # noinspection PyTypeChecker
            plt.fill_between(xplot,
                             gp.mu_new(xplot, gp.x_seen, gp.y_seen) - gp.var_new(xplot, gp.x_seen),
                             gp.mu_new(xplot, gp.x_seen, gp.y_seen) + gp.var_new(xplot, gp.x_seen),
                             color=gp.plot_col,
                             alpha=0.4)

            # Plot the "seen" points
            plt.scatter(gp.x_seen, gp.true_func(gp.x_seen), color="black", marker=".", label=f"{i}: Observed points",
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
