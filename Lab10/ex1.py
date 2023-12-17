import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
from multiprocessing import freeze_support

def main():

    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('C:\\Users\\Daniel\\Desktop\\PMP-2023\\Lab10\\dumy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    #a
    with pm.Model() as model_p:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = alfa + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    posterior_predictive = pm.sample_posterior_predictive(idata_p, model=model_p)
    az.plot_posterior(posterior_predictive['y_pred'], hdi_prob=0.95, color='lightblue')

    plt.scatter(x_1s, y_1s, label='Date reale', color='red')
    plt.plot(x_1s, posterior_predictive['y_pred'].mean(axis=0), label='Curba estimată', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Estimarea curbei cu un polinom de ordin 5')
    plt.legend()
    plt.show()

    #b
    with pm.Model() as model_p:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = alfa + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    posterior_predictive = pm.sample_posterior_predictive(idata_p, model=model_p)
    az.plot_posterior(posterior_predictive['y_pred'], hdi_prob=0.95, color='lightblue')

    plt.scatter(x_1s, y_1s, label='Date reale', color='red')
    plt.plot(x_1s, posterior_predictive['y_pred'].mean(axis=0), label='Curba estimată', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Estimarea curbei cu un polinom de ordin 5')
    plt.legend()
    plt.show()

    with pm.Model() as model_p:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = alfa + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
        
    posterior_predictive = pm.sample_posterior_predictive(idata_p, model=model_p)
    az.plot_posterior(posterior_predictive['y_pred'], hdi_prob=0.95, color='lightblue')

    plt.scatter(x_1s, y_1s, label='Date reale', color='red')
    plt.plot(x_1s, posterior_predictive['y_pred'].mean(axis=0), label='Curba estimată', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Estimarea curbei cu un polinom de ordin 5')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()