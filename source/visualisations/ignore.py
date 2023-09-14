import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start=0, stop=10, num=300)
plt.scatter(x, x+2, c=x+2, cmap='viridis',)
plt.show()