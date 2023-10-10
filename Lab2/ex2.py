import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

server1 = stats.gamma(4, 0, 1/3)
server2 = stats.gamma(4, 0, 1/2)
server3 = stats.gamma(5, 0, 1/2)
server4 = stats.gamma(5, 0, 1/3)
latenta = stats.expon(0, 1/4)

ps1 = 0.25
ps2 = 0.25
ps3 = 0.30
ps4 = 1 - ps1 - ps2 - ps3

nr_simulari = 10000
timp_procesare = np.zeros(nr_simulari)

for i in range(nr_simulari):
    server_selectat = np.random.choice([1, 2, 3, 4], p=[ps1, ps2, ps3, ps4])
    
    if server_selectat == 1:
        timp_procesare_server = server1.rvs()
    elif server_selectat == 2:
        timp_procesare_server = server2.rvs()
    elif server_selectat == 3:
        timp_procesare_server = server3.rvs()
    else:
        timp_procesare_server = server4.rvs()
    
    latență = latenta.rvs()
    timp_total_servire = timp_procesare_server + latență
    timp_procesare[i] = timp_total_servire

X_mai_mare_de_3 = np.mean(timp_procesare > 3)

print("Probabilitatea ca timpul de procesare sa fie mai mare de 3: ", X_mai_mare_de_3)

az.plot_posterior({'X':timp_procesare})
plt.show()