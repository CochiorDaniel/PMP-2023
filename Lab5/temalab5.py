import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

#ex1
model_trafic = pm.Model()

data = pd.read_csv("C:\\Users\\Daniel\\Desktop\\PMP-2023\\Lab5\\trafic.csv")

# deoarece inregistrarile incep de la ora 4:00
ora_7 = [(7-4)*60, (8-4)*60]        
ora_16 = [(16-4)*60, (17-4)*60]


ora_8 = [(8-4)*60, (9-4)*60]
ora_19 = [(19-4)*60, (20-4)*60]  

with model_trafic:
    lambda_init = pm.Exponential('lambda_init', lam=1)

    crestere = pm.Exponential('crestere', lam=1)
    descrestere = pm.Exponential('descrestere', lam=1)
    
    lambda_trafic = []
    for i, row in data.iterrows():
        if i in ora_7 or i in ora_16:
            lambda_i = lambda_init + crestere
        elif i in ora_8 or i in ora_19:
            lambda_i = lambda_init - descrestere
        else:
            lambda_i = lambda_init
        lambda_trafic.append(lambda_i)


    mean_lambda_trafic = np.mean(lambda_trafic)
    trafic = pm.Poisson('trafic', mu=mean_lambda_trafic, observed=data['nr. masini'])

with model_trafic:
    trace = pm.sample(20000)

pm.summary(trace)

#ex2
lambda_precedent = trace['lambda_init']
numar_intervale = 5

capete_intervale_probabile = []
valori_lambda_interval = []

for i in range(numar_intervale):
    inceput_interval = int(np.percentile(lambda_precedent[:, :], (i / numar_intervale) * 100))
    sfarsit_interval = int(np.percentile(lambda_precedent[:, :], ((i + 1) / numar_intervale) * 100))
    capete_intervale_probabile.append((inceput_interval, sfarsit_interval))

for interval in capete_intervale_probabile:
    medie_lambda = lambda_precedent[:, interval[0]:interval[1]].mean(axis=0)
    valori_lambda_interval.append(medie_lambda)

for i, interval in enumerate(capete_intervale_probabile):
    print(f"Intervalul de timp: {interval[0] // 60}:{interval[0] % 60} - {interval[1] // 60}:{interval[1] % 60}")
    print(f"Valori medii ale lui λ: {valori_lambda_interval[i]}")
    print()