import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

#1
data = pd.read_csv("C:\\Users\\Daniel\\Desktop\\PMP-2023\\Lab9\\Admission.csv")
x1 = data['GRE'].values.astype(int)
x2 = data['GPA'].values.astype(float)
admission = data['Admission'].values.astype(int)

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)

    pi = pm.Deterministic('pi', pm.math.sigmoid(beta0 + beta1*x1 + beta2*x2))

    admission_obs = pm.Bernoulli('admission_obs', p=pi, observed=admission)

with logistic_model:
    trace = pm.sample(2000, tune=1000)

pm.summary(trace)

#2
gre_values = np.linspace(data['GRE'].min(), data['GRE'].max(), 100)
gpa_values = np.linspace(data['GPA'].min(), data['GPA'].max(), 100)
x_grid = np.meshgrid(gre_values, gpa_values)
p_grid = pm.math.sigmoid(np.mean(trace['beta0']) + np.mean(trace['beta1']) * x_grid[0] + np.mean(trace['beta2']) * x_grid[1])
decision_boundary = np.mean(p_grid > 0.5)

hdi = az.hdi(trace['p'], hdi_prob=0.94)
print(f"Granita de decizie: {decision_boundary}")

p_grid_values = p_grid.eval()
decision_boundary_grid = p_grid_values > 0.5
hdi_grid = az.hdi(p_grid_values, hdi_prob=0.94)
plt.scatter(data['GRE'], data['GPA'], c=[f'C{x}' for x in data['Admission']])
plt.contour(x_grid[0], x_grid[1], decision_boundary_grid, levels=[0.5], colors='k')
plt.contourf(x_grid[0], x_grid[1], hdi_grid, levels=[hdi_grid.min(), 0.5, hdi_grid.max()], colors=['b', 'r', 'b'], alpha=0.5)
plt.xlabel('GRE')
plt.ylabel('GPA')
plt.legend(['Decision boundary', '94% HDI'])
plt.show()

#3
student1 = np.array([[550, 3.5]])
p_student1 = pm.math.sigmoid(np.mean(trace['beta0']) + np.mean(trace['beta1']) * student1[0, 0] + np.mean(trace['beta2']) * student1[0, 1]).eval()
hdi_student1 = az.hdi(p_student1, hdi_prob=0.9)
print(f"90% HDI for a student with GRE=550 and GPA=3.5: {hdi_student1[0]}, {hdi_student1[1]}")

#4
student2 = np.array([[500, 3.2]])
p_student2 = pm.math.sigmoid(np.mean(trace['beta0']) + np.mean(trace['beta1']) * student2[0, 0] + np.mean(trace['beta2']) * student2[0, 1]).eval()
hdi_student2 = az.hdi(p_student2, hdi_prob=0.9)
print(f"90% HDI for a student with GRE=500 and GPA=3.2: {hdi_student2[0]}, {hdi_student2[1]}")
