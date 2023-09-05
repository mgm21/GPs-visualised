import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.font_manager as fm

font_used = "Charter"
font_size = 25
font = {'fontname': font_used}
legend_font = fm.FontProperties(family=font_used)
legend_font._size = font_size

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
yarr = np.linspace(-4, 4, 100)

xloc1 = 0
mean1 = 0
std1 = 1

ax1.plot(stats.norm.pdf(yarr, mean1, std1), yarr, color="cornflowerblue")
ax2.plot(yarr, stats.norm.pdf(yarr, mean1, std1), color="cornflowerblue")

ax1.set_xlabel("x", fontsize=font_size, **font)
ax1.set_ylabel("f", fontsize=font_size, **font)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)

ax2.set_xlabel("x", fontsize=font_size, **font)
ax2.set_ylabel("f", fontsize=font_size, **font)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

fig.savefig("basic_normal_distribution", dpi=400, bbox_inches='tight')
plt.show()
