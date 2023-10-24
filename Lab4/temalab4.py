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

#Ex2
def calculeaza_probabilitate(alfa, num_simulari):
    val_bune = 0

    for _ in range(num_simulari):
        numar_clienti = np.random.poisson(lambda_param)
        timp_plasare_plata = np.random.normal(media_normal, deviatie_standard_normal, numar_clienti)
        timp_pregatire_comanda = np.random.exponential(alfa, numar_clienti)
        timp_total_servire = sum(timp_plasare_plata + timp_pregatire_comanda)/60 #convertim in minute
        #print("Timpul total de servire este: ", timp_total_servire)

        if timp_total_servire <= 15:
            val_bune += 1

    probabilitate = val_bune / num_simulari
    #print("Probabilitatea este: ", probabilitate)
    return probabilitate

max_alfa = 999
alfa = 100

while alfa > 0:
    p = calculeaza_probabilitate(alfa, 1000)
    if p >= 0.95:
        max_alfa = alfa
        break
    alfa -= 0.1

print("Valoarea maxima a lui alfa este: ", max_alfa)

#Ex3
alfa = max_alfa
timp_pregatire_comanda = np.random.exponential(alfa, numar_clienti)
timp_total_servire = sum(timp_plasare_plata + timp_pregatire_comanda)

timp_mediu_asteptare = timp_total_servire / numar_clienti
print("Timpul mediu de asteptare este: ", timp_mediu_asteptare)