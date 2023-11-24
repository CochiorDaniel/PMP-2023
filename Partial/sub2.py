import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats

#ex1
miu = 0
sigma = 1

timp_asteptare = stats.norm.rvs(miu, sigma, size=200)

#ex2
with pm.Model() as model:
   miu_initial = pm.Exponential('miu_initial', lam=1)
   sigma_initial = pm.Exponential('sigma_initial', lam=1)

 