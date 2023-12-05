import pandas as pd
import pymc as pm

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