import numpy as np

#Ex1
lambda_param = 20  
media_normal = 2  
deviatie_standard_normal = 0.5 
alfa = 5  #ales aleatoriu initial

numar_clienti = np.random.poisson(lambda_param)
timp_plasare_plata = np.random.normal(media_normal, deviatie_standard_normal, numar_clienti)
timp_pregatire_comanda = np.random.exponential(alfa, numar_clienti)
timp_total_servire = sum(timp_plasare_plata + timp_pregatire_comanda)
