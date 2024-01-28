import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():
    #1 ===============================================
    data = pd.read_csv("C:\\Users\\Daniel\\Desktop\\PMP-2023\\Lab8\\Prices.csv")
    x1 = data['Speed'].values.astype(float)
    x2 = np.log(data['HardDrive'].values.astype(float))
    y = data['Price'].values.astype(float)

    with pm.Model() as model_pret:
        alfa_tmp = pm.Normal('alfa_tmp', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)  #distributii slab informative
        epsilon = pm.HalfCauchy('epsilon', 5)

        miu = pm.Deterministic('miu', alfa_tmp + beta1 * x1 + beta2 * x2)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y) 

        trace = pm.sample(2000, tune=2000, return_inferencedata=True) #idata, adica ce generez dupa ce rulez modelul
                                                                    # trace e ce imi returneaza modelul dupa ce rulez sample

    az.plot_trace(trace, var_names=['alfa_tmp', 'beta1', 'beta2', 'epsilon'])
    plt.show()

    #2 ===============================================
    estimari = az.summary(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95) #rezumatul estimarilor
    print(estimari)

    #3 ===============================================
    #intervalul de incredere hdi 2.5% - 97.5%
    beta1_hdi_exclude_zero = 0 not in estimari['hdi_2.5%']['beta1'] and 0 not in estimari['hdi_97.5%']['beta1']
    beta2_hdi_exclude_zero = 0 not in estimari['hdi_2.5%']['beta2'] and 0 not in estimari['hdi_97.5%']['beta2']
    #sau
    # is_beta1_significant = (summary['hdi_2.5%']['beta1'] > 0) or (summary['hdi_97.5%']['beta1'] < 0)
    # is_beta2_significant = (summary['hdi_2.5%']['beta2'] > 0) or (summary['hdi_97.5%']['beta2'] < 0)

    # daca se afla in intervalul de incerdere, atunci beta1 si beta2 sunt predictori utili

    print("Frecvența procesorului este un predictor util:", beta1_hdi_exclude_zero)
    print("Mărimea hard diskului este un predictor util:", beta2_hdi_exclude_zero)

    #4 ===============================================
    x1_nou = 33
    x2_nou = np.log(540)
    miu_nou = trace['alfa_tmp'] + trace['beta1']*x1_nou + trace['beta2']*x2_nou

    pret_asteptat = np.random.choice(miu_nou, size=5000)
    hdi = az.hdi(pret_asteptat, hdi_prob=0.9)
    print("Intervalul de 90% HDI pentru prețul așteptat este:", hdi)

    #5 ===============================================
    pret_simulat = np.random.normal(loc=miu_nou, scale=trace['epsilon'], size=5000)

    hdi_pred = az.hdi(pret_simulat, hdi_prob=0.9)
    print("Intervalul de predicție de 90% HDI pentru prețul de vânzare este:", hdi_pred)

    #sau
    # ppc = pm.sample_posterior_predictive(trace, model=model_pret)
    # posterior_predictive = ppc['y_pred']

    # hdi_90_posterior_predictive = az.hdi(posterior_predictive.flatten(), hdi_prob=0.9)
    # print(f"Intervalul de 90% HDI pentru distribuția predictivă posterioară este: "
    #           f"[{hdi_90_posterior_predictive[0]:.2f}, {hdi_90_posterior_predictive[1]:.2f}]")

if __name__ == '__main__':
    freeze_support()
    main()