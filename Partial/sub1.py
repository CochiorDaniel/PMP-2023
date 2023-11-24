import random
import numpy as np
from scipy import stats
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


#ex1

stema_moneda_P0 = 1/3
stema_moneda_P1 = 1/2
jucatori = { 0:0, 1:0}

def aruncare_moneda_start():
    return random.choice([0, 1])

def stema(probabilitate):
    return random.choices([0, 1], [1 - probabilitate, probabilitate]) # 1 este stema

for i in range(20000):
    primul_jucator = aruncare_moneda_start()
    if primul_jucator == 0:
        n = stema(stema_moneda_P0)
        m = stema(stema_moneda_P1) + stema(stema_moneda_P1)
    else:
        n = stema(stema_moneda_P1)
        m = stema(stema_moneda_P0) + stema(stema_moneda_P0)

    if n >= m:
        jucatori[primul_jucator] += 1
    else:
        jucatori[1 - primul_jucator] += 1


print("Sanse P0: ", jucatori[0] / 20000)
print("Sanse P1: ", jucatori[1] / 20000)
#Dupa rulare, se observa ca P1 are sanse mai mari de castig

#ex2
model = BayesianNetwork([('P0', 'win'), ('P1', 'win')])

P0_stema = TabularCPD('P0', 2, [[1 - stema_moneda_P0, stema_moneda_P0]])
P1_stema = TabularCPD('P1', 2, [[1 - stema_moneda_P1, stema_moneda_P1]])

win = TabularCPD('win', 2, 
                    [[1/2,1/4,1/2,3/4], 
                     [1/2,3/4,1/2,1/4]],
                 evidence=['P0', 'P1'], 
                 evidence_card=[2, 2])

model.add_cpds(P0_stema, P1_stema, win)
