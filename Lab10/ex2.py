import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from multiprocessing import freeze_support

az.style.use('arviz-darkgrid')

def main():

    np.random.seed(42)
    x_1_large = np.linspace(-2, 2, 500)
    y_1_large = 2 * x_1_large**5 - 4 * x_1_large**4 + 3 * x_1_large**3 + np.random.normal(0, 1, size=500)

    order = 5
    x_1p_large = np.vstack([x_1_large**i for i in range(1, order + 1)])
    x_1s_large = (x_1p_large - x_1p_large.mean(axis=1, keepdims=True)) / x_1p_large.std(axis=1, keepdims=True)
    y_1s_large = (y_1_large - y_1_large.mean()) / y_1_large.std()

    # Model cu sd=10
    with pm.Model() as model_p_large:
        alfa_large = pm.Normal('alfa', mu=0, sigma=1)
        β_large = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε_large = pm.HalfNormal('ε', 5)
        μ_large = alfa_large + pm.math.dot(β_large, x_1s_large)
        y_pred_large = pm.Normal('y_pred', mu=μ_large, sigma=ε_large, observed=y_1s_large)
        idata_p_sd_10_large = pm.sample(2000, return_inferencedata=True)

    # Model cu sd=100
    with pm.Model() as model_p_sd_100_large:
        alfa_large = pm.Normal('alfa', mu=0, sigma=1)
        β_large = pm.Normal('β', mu=0, sigma=100, shape=order)
        ε_large = pm.HalfNormal('ε', 5)
        μ_large = alfa_large + pm.math.dot(β_large, x_1s_large)
        y_pred_large = pm.Normal('y_pred', mu=μ_large, sigma=ε_large, observed=y_1s_large)
        idata_p_sd_100_large = pm.sample(2000, return_inferencedata=True)

    # Model cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
    with pm.Model() as model_p_sd_array_large:
        alfa_large = pm.Normal('alfa', mu=0, sigma=1)
        β_large = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε_large = pm.HalfNormal('ε', 5)
        μ_large = alfa_large + pm.math.dot(β_large, x_1s_large)
        y_large = pm.Normal('y_pred', mu=μ_large, sigma=ε_large, observed=y_1s_large)
        idata_p_sd_array_large = pm.sample(2000, return_inferencedata=True)

    plt.figure(figsize=(10, 10))

    # Grafic pentru sd=10
    plt.subplot(3, 1, 1)
    plt.scatter(x_1s_large[0], y_1s_large)
    az.plot_posterior(idata_p_sd_10_large)
    plt.title('Model cu sd=10 (500 de puncte)')

    # Grafic pentru sd=100
    plt.subplot(3, 1, 2)
    plt.scatter(x_1s_large[0], y_1s_large)
    az.plot_posterior(idata_p_sd_100_large)
    plt.title('Model cu sd=100 (500 de puncte)')

    # Grafic pentru sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
    plt.subplot(3, 1, 3)
    plt.scatter(x_1s_large[0], y_1s_large)
    az.plot_posterior(idata_p_sd_array_large)
    plt.title('Model cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1]) (500 de puncte)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()