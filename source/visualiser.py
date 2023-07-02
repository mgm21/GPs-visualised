import matplotlib.pyplot as plt


class Visualiser:
    def __init__(self):
        self.plot_cols = ["cornflowerblue", "seagreen", "tan", "lightcoral", "mediumorchid"]
        self.show_axes_ticks_labels = True
        
    def plot_gps_matplotlib(self, gps_arr, savefig=True):
        # Set up the plotting environment
        plt.figure()
        
        for i, gp in enumerate(gps_arr):

            gp.plot_col = self.plot_cols[i]

            xplot = gp.x_problem

            # Plot the true, hidden, function
            plt.plot(xplot, gp.true_func(xplot), color=gp.plot_col, linestyle="--", label=f"{i}: True function", zorder=1)
    
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