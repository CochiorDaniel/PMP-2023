import pymc3 as pm
import arviz as az

Y_val = [0, 5, 10]
theta_val = [0.2, 0.5]
results = []

for Y in Y_val:
    for theta in theta_val:
        with pm.Model() as model:
            a_priori_n = pm.Poisson('a_priori_n', mu=10)
            Y_obs = pm.Binomial('Y_obs', n=a_priori_n, p=theta, observed=Y)
            a_posteriori_n = pm.sample(2000, tune=1000, cores=2)
        
        results.append(a_posteriori_n)

for result in results:
    az.plot_posterior(result)
