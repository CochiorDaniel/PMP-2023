import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

probabilitate_primul_mecanic = 0.4
probabilitate_al_doilea_mecanic = 1 - probabilitate_primul_mecanic

mec1 = stats.expon(0, 1/4).rvs(10000)
mec2 = stats.expon(0, 1/6).rvs(10000)

X = probabilitate_primul_mecanic*mec1 + probabilitate_al_doilea_mecanic*mec2

media = np.mean(X)
deviatia_standard = np.std(X)

print("Media: ", media)
print("Deviatia standard: ", deviatia_standard)

az.plot_posterior({'X':X})
plt.show()