import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az


#a  
data = pd.read_csv("C:\\Users\\Daniel\\Desktop\\PMP-2023\\Lab7\\auto-mpg.csv")
data.dropna(inplace=True)

mpg=data['mpg'].values
CP=data['horsepower'].values

plt.scatter(CP, mpg, color='blue')
plt.xlabel('CP')
plt.ylabel('MPG')

#b
model_consum = pm.Model()
CP = pd.to_numeric(CP, errors='coerce')

with model_consum:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    std_dev = pm.HalfCauchy('std_dev', 5)
 
    
    y = alpha + beta * CP
    consum_prezis = pm.Normal('mpg', mu=y, sigma=std_dev, observed=mpg)
    
    trace = pm.sample(2000, tune=2000, return_inferencedata=True)
    az.plot_trace(trace, var_names=['alpha', 'beta', 'std_dev'])

#c
posterior_g = trace.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['a'].mean().item()
beta_m = posterior_g['β'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(CP, posterior_g['a'][draws].values+ posterior_g['β'][draws].values * CP[:,None],
c='gray', alpha=0.5)
plt.plot(CP, alpha_m + beta_m * CP, c='k',label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('CP')
plt.ylabel('MPG')
 
#d
plt.plot(CP, alpha_m + beta_m * CP, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
sig = az.plot_hdi(CP, posterior_g['μ'].T, hdi_prob=0.95, color='k')
plt.xlabel('x')
plt.ylabel('y', rotation=0)

plt.show()