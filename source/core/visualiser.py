import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, State, callback
from source.adapters.gpcf import GPCF
import plotly.express as px
import pandas as pd
import ipywidgets


class Visualiser:
    def __init__(self):
        # Library-agnostic parameters
        self.plot_cols = ["cornflowerblue", "tan", "seagreen", "mediumorchid", "silver", "salmon", "blue",
                          "green", "red"]

        # Plotly-specific
        self.num_plotly_objects_per_gp = 5

        # Matplotlib-specific
        self.show_axes_ticks_labels = True

    def plot_gps_matplotlib(self,
                            gps_arr,
                            savefig=True,
                            plot_elements=np.array(["true", "mean", "var", "observed", "acquisition"])):
        plt.figure()

        for i, gp in enumerate(gps_arr):

            gp.plot_col = self.plot_cols[i]

            xplot = gp.x_problem

            if "true" in plot_elements:
                # Plot the true, hidden, function
                plt.plot(xplot, gp.true_func(xplot), color=gp.plot_col, linestyle="--", label=f"{i}: True function",
                         zorder=1)

            if "mean" in plot_elements:
                # Plot the updated GP mean
                plt.plot(xplot, gp.mu_new(xplot), color=gp.plot_col, label=f"{i}: GP mean", zorder=1)

            if "var" in plot_elements:
                # Plot the uncertainty bands
                plt.fill_between(xplot,
                                 gp.mu_new(xplot) - gp.var_new(xplot),
                                 gp.mu_new(xplot) + gp.var_new(xplot),
                                 color=gp.plot_col,
                                 alpha=0.4)

            if "observed" in plot_elements:
                # Plot the seen points
                plt.scatter(gp.x_seen, gp.true_func(gp.x_seen),
                            color="black", marker=".", label=f"{i}: Observed points",
                            zorder=2, linewidths=1, s=40,
                            alpha=0.4)

            plt.legend()

            # Edit plot layout
            if not self.show_axes_ticks_labels:
                plt.tick_params(left=False, right=False, labelleft=False,
                                labelbottom=False, bottom=False)

        if savefig:
            plt.savefig("example-sin", dpi=20)

        plt.show()

    def visualise_gps_plotly(self,
                             gps_arr,
                             plot_elements=np.array(["true", "mean", "var", "observed", "acquisition"])):
        # Based on last part of tutorial: https://www.youtube.com/watch?v=pNMWbY0AUJ0&t=1531s
        initial_fig = self._generate_plotly_figure(gps_arr, plot_elements)

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
            fig = self._generate_plotly_figure(gps_arr, plot_elements)

            print(point_clicked)
            print(type(point_clicked))
            print(x)

            return fig

        app.run(port=8000)

    def visualise_ite_plotly(self,
                             gp,
                             plot_elements=np.array(["true", "mean", "var", "observed", "acquisition"])):
        initial_fig = self._generate_plotly_figure([gp], plot_elements)
        initial_fig = self._plot_end_cond_thresh(initial_fig, gp)
        initial_fig.update_layout(
            title=dict(text="ITE Algorithm", font=dict(size=50), automargin=False, yref='paper'),
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
            fig = self._generate_plotly_figure([gp], plot_elements)
            fig = self._plot_end_cond_thresh(fig, gp)
            fig.update_layout(
                title=dict(text="ITE Algorithm", font=dict(size=50), automargin=False, yref='paper'),
                title_x=0.5
            )

            # print(point_clicked)
            # print(type(point_clicked))
            # print(x)

            return fig

        app.run(port=8001)

    def visualise_gpcf_plotly(self,
                              gps_arr,
                              plot_elements=np.array(["mean", "observed"])):

        # Based on last part of tutorial: https://www.youtube.com/watch?v=pNMWbY0AUJ0&t=1531s

        initial_fig = self._generate_plotly_figure(gps_arr, plot_elements)

        initial_fig.update_layout(
            title=dict(text="GPCF Algorithm", font=dict(size=50), automargin=False, yref='paper'),
            title_x=0.5
        )

        # different line than visualise example experiment
        # Initialise GPCF adapter
        gpcf = GPCF(gps_arr[:-1], gps_arr[-1])

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

            gps_arr[gp_index].update_gp(x_clicked=x)

            # different line than visualise example experiment
            gpcf.update_current_gp_mu_0()

            fig = self._generate_plotly_figure(gps_arr, plot_elements)

            initial_fig.update_layout(
                title=dict(text="GPCF Algorithm", font=dict(size=50), automargin=False, yref='paper'),
                title_x=0.5
            )

            print(point_clicked)
            print(type(point_clicked))
            print(x)

            return fig

        app.run(port=8002)

    def _plot_end_cond_thresh(self,
                              fig,
                              gp):
        thresh = np.repeat(a=gp.calculate_end_cond_thresh_val(), repeats=len(gp.x_problem))

        fig.add_scatter(x=gp.x_problem,
                        y=thresh,
                        line=dict(color="red", dash='dash', width=4),
                        name=f"End condition")

        return fig

    def _generate_plotly_figure(self,
                                gps_arr,
                                plot_elements,
                                fig=go.FigureWidget()):

        for i, gp in enumerate(gps_arr):
            # Compute/Define the required arrays once
            gp.plot_col = self.plot_cols[i]
            mu_new = gp.mu_new(gp.x_problem)
            var_new = gp.var_new(gp.x_problem)
            xplot = gp.x_problem
            y_upper = mu_new + var_new
            y_lower = mu_new - var_new
            # The suggested x location is cast to a numpy array because mu_0 must receive array to behave as intended
            acquisition_x_suggestion = np.array([gp.query_acquisition_function()])

            # Note: ordering is important for click events (last Scatter added is never covered by others)
            # Note: make sure that the number of objects below matches self.num_plotly_objects_per_gp attribute

            if "var" in plot_elements:
                # GP uncertainty
                fig.add_scatter(x=np.append(xplot, xplot[::-1]),
                                y=np.append(y_upper, y_lower[::-1]),
                                fill='toself',
                                line=dict(color=gp.plot_col),
                                name=f"{i} GP uncertainty"),

            if "mean" in plot_elements:
                # GP mean
                fig.add_scatter(x=xplot,
                                y=mu_new,
                                line=dict(color=gp.plot_col),
                                name=f"{i} GP mean"),

            if "true" in plot_elements:
                # True function
                fig.add_scatter(x=xplot,
                                y=gp.true_func(xplot),
                                line=dict(color=gp.plot_col, dash='dash', width=4),
                                name=f"{i} True function"),

            if "observed" in plot_elements:
                # Observed points
                fig.add_scatter(x=gp.x_seen,
                                y=gp.true_func(gp.x_seen),
                                mode="markers",
                                name=f"{i} Observed points",
                                marker=dict(color=gp.plot_col, size=10),
                                opacity=0.5),

            if "acquisition" in plot_elements:
                # Acquisition function suggestion
                fig.add_scatter(x=acquisition_x_suggestion,
                                y=gp.mu_new(acquisition_x_suggestion),
                                marker=dict(color="black", size=10, symbol="cross"),
                                mode="markers",
                                opacity=0.5,
                                name=f"{i} Acquisition suggestion")

        return fig

# BACKLOG

# Todo: put an option to plot reduced curves (it is too crammed right now). For instance gpcf only gets ancestor
#  means, no point plotting uncertainty and acquisition suggestion (even potentially true function, though that
#  helps). Could pass in true function as initially hidden. Maybe make that initially hidden and that it
#  can be toggled. That way, you can have all the information at hand, but it is easy to see in the default setting.

# Todo: check how to update the plots vs re-plotting them at each click

# Todo: I rarely (irreproducibly) got singular matrix error when following through with what the acquisition wanted,
#  this led to very spiky uncertainty bands. Should I check whether the matrix is singular?

# Todo: include an animation component whereby the optimisation can be done automatically with a sleep call in
#  between re-plotting. add a play button to do the whole adaptation process for you without having to click etc...
#  and this is where the sleep(1) will come in handy.

# Todo: most of the public methods can be collapsed into one with an argument passed to select an experiment/algo
