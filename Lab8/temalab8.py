import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

#1
data = pd.read_csv("C:\\Users\\Daniel\\Desktop\\PMP-2023\\Lab8\\Prices.csv")
x1 = data['Speed'].values.astype(float)
x2 = np.log(data['HardDrive'].values.astype(float))
y = data['Price'].values.astype(float)

with pm.Model() as model_pret:
    alfa_tmp = pm.Normal('alfa_tmp', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=1)
    beta2 = pm.Normal('beta2', mu=0, sigma=1)
    epsilon = pm.HalfCauchy('epsilon', 5)

    miu = pm.Deterministic('miu', alfa_tmp + beta1 * x1 + beta2 * x2)
    y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y)

    trace = pm.sample(2000, tune=2000, return_inferencedata=True)

az.plot_trace(trace, var_names=['alfa_tmp', 'beta1', 'beta2', 'epsilon'])
plt.show()

#2
estimari = az.summary(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)
print(estimari)

#3
beta1_hdi_exclude_zero = 0 not in estimari['hdi_2.5%']['beta1'] and 0 not in estimari['hdi_97.5%']['beta1']
beta2_hdi_exclude_zero = 0 not in estimari['hdi_2.5%']['beta2'] and 0 not in estimari['hdi_97.5%']['beta2']
print("Frecvența procesorului este un predictor util:", beta1_hdi_exclude_zero)
print("Mărimea hard diskului este un predictor util:", beta2_hdi_exclude_zero)


