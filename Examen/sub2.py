import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#a
def posterior_grid(grid_points=50, heads=6, tails=9, prima_stema=5):

    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) 

    # probabilitatea de a obține prima stema la o aruncare
    likelihood = np.zeros(grid_points)
    likelihood[prima_stema - 1] = 1.0

    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

# aparitia primei steme la aruncarea a 5a aruncare, atunci cand a=1
data = np.repeat([0, 0, 0, 0, 1], (4, 1, 1, 1, 1))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t, prima_stema=5)

plt.plot(grid, posterior, 'o-')
plt.title(f'Prima stema la aruncarea a 5 - stema = {h}, pajura = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

#b
theta_max_posterior = grid[np.argmax(posterior)]
print(f'Estimarea lui θ care maximizează probabilitatea a posteriori: {theta_max_posterior:.2f}')


