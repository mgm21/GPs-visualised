from source.core.gaussian_process import GaussianProcess
from source.core.visualiser import Visualiser
from source.agents.example_agents import agents

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

visualiser = Visualiser()

plt.style.use("seaborn")
fig, ax = plt.subplots()

child = [agents[0],] # Must be a list
ancestors = [agents[1:3],] # Must be a list

visualiser.update_gps_axes_matplotlib(ax=ax, gps_arr=child,)

# Plot config
font_used = "Charter"
font_size = 21
font = {'fontname': font_used}
legend_font = fm.FontProperties(family=font_used)
legend_font._size = font_size
ax.set_ylim(-6, 6)
ax.set_xlim(0, 10)
ax.set_xlabel("Behavioural descriptor", fontsize=font_size, **font)
ax.set_ylabel("Fitness", fontsize=font_size, **font)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(prop=legend_font, bbox_to_anchor=(1.05, 1), loc='upper left') # For legend outside the plot
# ax.legend(prop=legend_font,) # For legend in the plot
fig.savefig("my-plot.png", dpi=600, bbox_inches='tight')
plt.show()
