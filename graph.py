from matplotlib import pyplot as plt
import numpy as np
import random

rng = np.random.default_rng(4567)

# Generate data
x = rng.uniform(0, 10, size=10)
y = x + rng.normal(size=10)

# Initialize layout
fig, ax = plt.subplots(figsize=(9, 9))

# Add scatterplot
ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

# Fit linear regression via least squares with numpy.polyfit
# It returns an slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
b, a = np.polyfit(x, y, deg=1)

# Create sequence of 100 numbers from 0 to 100
xseq = np.linspace(0, 10, num=100)

# Plot regression line
ax.plot(xseq, a + b * xseq, color="k", alpha=0, lw=2.5)

plt.savefig('pointplot.png', dpi=600)
plt.show()
