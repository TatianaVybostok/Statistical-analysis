import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import InterpolatedUnivariateSpline, Rbf
from scipy import interpolate
import random
from scipy import stats



#kindex
f4 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindexfiltr.txt', 'r')
lines4 = f4.readlines()
f4.close()
y=[]
for line in lines4:
        p = line.split()
        y.append(float(p[0]))
yv = np.array(y)

#pocet poruch        
f5 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/pocetporuch.txt', 'r')
lines5 = f5.readlines()
f5.close()

a=[]
for line in lines5:
        p = line.split()
        a.append(float(p[0]))
av = np.array(a)

#plt.plot(yv,av[1:])
#plt.show()


N_h=[49,39,36,45,82,105,113,133,139,131]
N_h=sum(N_h)
N_l=[39,42,33,32,36,33,22,36,35,33]
N_l=sum(N_l)

N_hm=[11,10,8,4,16,20,25,28,30,32]
N_hm=sum(N_hm)
N_lm=[7,5,4,7,9,7,4,6,6,7]
N_lm=sum(N_lm)

exces=(N_hm-N_lm)/(N_hm+N_lm)*100 
exces_orig=(N_h-N_l)/(N_h+N_l)*100 

def percentage(part, whole):
      percentage = 100 * float(part)/float(whole)
      return str(percentage) + "%"

print(percentage(exces,exces_orig)) #113.29