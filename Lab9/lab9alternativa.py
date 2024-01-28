import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv("C:\\Users\\Claudia\\Desktop\\an3\\PMP\\Admission.csv")

gre = data['GRE'].values
gpa = data['GPA'].values
admission = data['Admission'].values

df = data.query("Admission == ('0', '1')")
y_1 = pd.Categorical(df['Admission']).codes
x_n = ['GRE', 'GPA'] # multipla, poate fi si cu 1, am ex i curs
x_1 = df[x_n].values
# de pus cand am regresie logistica multipla

#1
if __name__ == '__main__':
    with pm.Model() as model_1:
        a = pm.Normal('a', mu=0, sigma=10)
        β = pm.Normal('β', mu=0, sigma=2, shape=len(x_n))
        μ = a + pm.math.dot(x_1, β)
        θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ)))
        bd = pm.Deterministic('bd', -a/β[1] - β[0]/β[1] * x_1[:,0])
        yl = pm.Bernoulli('yl', p=θ, observed=y_1)
        idata_1 = pm.sample(2000, return_inferencedata=True)

#2  
    idx = np.argsort(x_1[:, 0])
    bd = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]

    plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in y_1]) 
    plt.plot(x_1[:, 0][idx], bd, color='k')
    az.plot_hdi(x_1[:, 0], idata_1.posterior['bd'], color='k', hdi_prob=0.94)
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])
    plt.legend()
    plt.show()

#3
    date_student1 = np.array([[1, 550, 3.5]])

    posterior_probs_student1 = 1 / (1 + np.exp(-idata_1.posterior['a'].values - np.dot(date_student1[:, 1:], idata_1.posterior['β'].values.T)))
    hdi_prob = az.hdi(posterior_probs_student1, hdi_prob=0.9)
    print(f"Intervalul HDI pentru probabilitatea de admitere: [{hdi_prob[0]:.4f}, {hdi_prob[1]:.4f}]")

#4
    date_student2 = np.array([[1, 500, 3.2]])

    posterior_probs_student2 = 1 / (1 + np.exp(-idata_1.posterior['a'].values - np.dot(date_student2[:, 1:], idata_1.posterior['β'].values.T)))
    hdi_prob = az.hdi(posterior_probs_student2, hdi_prob=0.9)
    print(f"Intervalul HDI pentru probabilitatea de admitere: [{hdi_prob[0]:.4f}, {hdi_prob[1]:.4f}]")

# Diferenta se justifica prin faptul ca GRE si GPA pot afecta in mod diferit probabilitatea de a fi admis sau nu, modelul luand in calcul interactiunea dintre ele