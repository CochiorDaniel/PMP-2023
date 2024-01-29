import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():
    #a
    data = pd.read_csv("C:\\Users\\Daniel\\Desktop\\PMP-2023\\Examen\\BostonHousing.csv")
    nr_mediu_camere = data['rm'].values.astype(float)
    rata_criminalitatii = data['crim'].values.astype(float)
    suprafate_comerciale = data['indus'].values.astype(float)
    valoare = data['medv'].values.astype(float)

    #b
    with pm.Model() as model_pret:
        alfa = pm.Normal('alfa_tmp', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        beta3 = pm.Normal('beta3', mu=0, sigma=1)
        epsilon = pm.HalfCauchy('epsilon', 5)

        y_pred = pm.Deterministic('y_pred', alfa + beta1 * nr_mediu_camere + beta2 * rata_criminalitatii + beta3 * suprafate_comerciale)
        medv_pred = pm.Normal('medv_pred', mu=y_pred, sigma=epsilon, observed=valoare)

        trace = pm.sample(2000, tune=2000, return_inferencedata=True)

    #c
    az.plot_forest(trace, hdi_prob=0.95 , var_names=['alfa_tmp', 'beta1', 'beta2', 'beta3', 'epsilon'])
    az.summary(trace,  hdi_prob=0.95 ,var_names=['beta1', 'beta2', 'beta3'])
    plt.show()
    # Variabila care influenteaza cel mai mult rezultatul este nr_mediu_camere deoarece beta1 are valoarea cea mai mare,
    # in comparatie cu beta2 si beta3 care sunt aproape 0. Aceasta se poate vedea si in png-ul atasat pt acest subpunct.

    #d
    ppc = pm.sample_posterior_predictive(trace, model=model_pret)
    y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_ppc,hdi_prob=0.5)
    hdi_pred = az.hdi(ppc, hdi_prob=0.5)
    print("Intervalul de predicție de 50% HDI pentru valoarea locuintelor:", hdi_pred)



if __name__ == "__main__":
    freeze_support()
    main()