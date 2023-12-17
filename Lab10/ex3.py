import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

az.style.use('arviz-darkgrid')
np.random.seed(42)

def main():
    x_cubic = np.linspace(-2, 2, 500)
    y_cubic = x_cubic**3 + np.random.normal(0, 1, size=500)

    order_cubic = 3
    x_cubic_p = np.vstack([x_cubic**i for i in range(1, order_cubic + 1)])
    x_cubic_s = (x_cubic_p - x_cubic_p.mean(axis=1, keepdims=True)) / x_cubic_p.std(axis=1, keepdims=True)
    y_cubic_s = (y_cubic - y_cubic.mean()) / y_cubic.std()

    with pm.Model() as model_cubic:
        alfa_cubic = pm.Normal('alfa', mu=0, sigma=1)
        β_cubic = pm.Normal('β', mu=0, sigma=1, shape=order_cubic)
        ε_cubic = pm.HalfNormal('ε', 5)
        μ_cubic = alfa_cubic + pm.math.dot(β_cubic, x_cubic_s)
        y_pred_cubic = pm.Normal('y_pred', mu=μ_cubic, sigma=ε_cubic, observed=y_cubic_s)
        idata_cubic = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_linear:
        alfa_linear = pm.Normal('alfa', mu=0, sigma=1)
        β_linear = pm.Normal('β', mu=0, sigma=1, shape=1)
        ε_linear = pm.HalfNormal('ε', 5)
        #μ_linear = alfa_linear + pm.math.dot(β_linear, x_cubic_s[0])
        μ_linear = alfa_linear + β_linear *  x_cubic_s[0]
        y_pred_linear = pm.Normal('y_pred', mu=μ_linear, sigma=ε_linear, observed=y_cubic_s)
        idata_linear = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_quadratic:
        alfa_quadratic = pm.Normal('alfa', mu=0, sigma=1)
        β_quadratic = pm.Normal('β', mu=0, sigma=1, shape=2)
        ε_quadratic = pm.HalfNormal('ε', 5)
        μ_quadratic = alfa_quadratic + pm.math.dot(β_quadratic, x_cubic_s[:2])
        y_pred_quadratic = pm.Normal('y_pred', mu=μ_quadratic, sigma=ε_quadratic, observed=y_cubic_s)
        idata_quadratic = pm.sample(2000, return_inferencedata=True)

    plt.figure(figsize=(12, 8))

    waic_cubic = az.waic(idata_cubic)
    waic_linear = az.waic(idata_linear)
    waic_quadratic = az.waic(idata_quadratic)
    plt.subplot(2, 1, 1)
    az.plot_waic([waic_cubic, waic_linear, waic_quadratic], labels=['Cubic', 'Linear', 'Quadratic'])
    plt.title('WAIC Comparison')

    loo_cubic = az.loo(idata_cubic)
    loo_linear = az.loo(idata_linear)
    loo_quadratic = az.loo(idata_quadratic)
    plt.subplot(2, 1, 2)
    az.plot_loo([loo_cubic, loo_linear, loo_quadratic], labels=['Cubic', 'Linear', 'Quadratic'])
    plt.title('LOO Comparison')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()