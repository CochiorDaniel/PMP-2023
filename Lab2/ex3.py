import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

numar_experimente = 100
numar_aruncari = 10

probabilitate_stema = 0.3

rezultate = np.empty((numar_experimente, numar_aruncari), dtype=str)

for i in range(numar_experimente):
    aruncari_moneda1 = np.random.choice(['s', 'b'], size=numar_aruncari)
    aruncari_moneda2 = np.random.choice(['s', 'b'], size=numar_aruncari, p=[1 - probabilitate_stema, probabilitate_stema])
    rezultate_experiment = [moneda1 + moneda2 for moneda1, moneda2 in zip(aruncari_moneda1, aruncari_moneda2)]
    rezultate[i] = rezultate_experiment

numar_ss = np.sum(rezultate == 'ss', axis=1)
numar_sb = np.sum(rezultate == 'sb', axis=1)
numar_bs = np.sum(rezultate == 'bs', axis=1)
numar_bb = np.sum(rezultate == 'bb', axis=1)