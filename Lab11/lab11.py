import numpy as np
import arviz as az
import pymc as pm
from multiprocessing import freeze_support

def main():
    #1
    # generarea datelor
    # 500 de date din 3 distributii gausiene (orice distributie gausiana are media si deviatia standard)
    clusters = 3 # numarul de distributii gausiene
    n_cluster = [200, 130, 170] # numarul de puncte din fiecare distributie
    n_total = sum(n_cluster) # numarul total de puncte
    means = [5, 0, 2] # media fiecarei distributii
    std_devs = [2, 2, 2] # deviatia standard a fiecarei distributii
    mix = np.random.normal(np.repeat(means, n_cluster), 
    np.repeat(std_devs, n_cluster)) # distributia mixta
    az.plot_kde(np.array(mix)) 

    #2
    mix = np.array(mix)

    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                            mu=np.linspace(mix.min(), mix.max(), cluster),
                            sigma=10, shape=cluster,
                            transform=pm.distributions.transforms.ordered)
            sd = pm.HalfNormal('sd', sigma=100)
            y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
            idata = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)

    #3
    comp_waic = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="waic", scale="deviance")
    comp_loo = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="loo", scale="deviance")

    az.plot_compare(comp_waic)
    az.plot_compare(comp_loo)

if __name__ == "__main__":
    freeze_support()
    main()