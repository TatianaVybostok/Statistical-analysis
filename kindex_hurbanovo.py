from datetime import datetime,timedelta,date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,glob
import functools
from scipy.ndimage import gaussian_filter1d

k_list=[]

path_k = '/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/'
kh = [f for f in os.listdir(path_k) if f.endswith('.txt')]
#kh=['k_2009.txt']
kh=sorted(kh)
col_mt=['number','k1','k2','k3','k4','k5','k6','k7','k8']

for i,item in enumerate(kh): 

    filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/'+str(item)
    f = open(filein, 'r')
    lines = f.readlines()
    f.close()
    x=[]
    k1=[]
    k2=[]
    k3=[]
    k4=[]
    k5=[]
    k6=[]
    k7=[]
    k81=[]
    for i,line in enumerate(lines):
            #p = line.split(',')
            p = line.split()
            x.append((p[0])) #poradie
            k1.append((p[1])) #kindex    
            k2.append((p[2])) #kindex 
            k3.append((p[3])) #kindex 
            k4.append((p[4])) #kindex 
            k5.append((p[5])) #kindex 
            k6.append((p[6])) #kindex 
            k7.append((p[7])) #kindex 
            k81.append((p[8])) #kindex  
           
    k82=[]
    for i in k81:
        k82.append(i.replace("\n", ""))
    k83=[]
    for i in k82:
        k83.append(i.replace("\x1a", ""))


    number=list(map(int,x))
    k1= list(map(int, k1))
    k2= list(map(int, k2))
    k3= list(map(int, k3))
    k4= list(map(int, k4))
    k5= list(map(int, k5))
    k6= list(map(int, k6))
    k7= list(map(int, k7))
    k8= list(map(int, k83))

    #k.append([sum(x)/len(x) for x in zip(k1,k2,k3,k4,k5,k6,k7,k8)])
    k_list.append([(a+b+c+d+e+f+g+h) / 8 for a,b,c,d,e,f,g,h in zip(k1,k2,k3,k4,k5,k6,k7,k8)])
#create flat list from list of lists
k_h = [item for sublist in k_list for item in sublist]
poradie=range(1,len(k_h)+1)

res = [idx for idx, val in enumerate(k_h) if val > 5]
print(res)


#k_h=gaussian_filter1d(np.asarray(k_h).astype('float'),50)


sdate=date(year = 2009, month = 1, day = 1)
edate=date(year = 2019, month = 12, day = 31)
dt_k_h=pd.date_range(sdate,edate,freq='d').date


############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

#kindex
filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/BDV_K-index.txt'
f = open(filein, 'r')
lines = f.readlines()
f.close()
#x=[]
dates=[]
k=[]
for line in lines:
        p = line.split()
        dates.append((p[0])) #datum
        #dates.append(p[1]) #datum
        k.append(float(p[9])) #kindex                
#poradie = np.array(x)
k= np.array(k)
dt_k=[]
for i,item in enumerate(dates):
    dt_k.append(datetime.strptime(str(item), '%d.%m.%Y').date()) 

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
#for K INDEX AND METEODATA
first_ind=[]
last_ind=[]
for i,item in enumerate(dt_k):
    if item == sdate:
        first_ind=i
    elif item == edate:
        last_ind=i
    else:
        pass

#OREZANIE POLI        
dt_k=dt_k[first_ind:last_ind+1]
k=k[first_ind:last_ind+1]
print(dt_k[0],dt_k[-1],len(dt_k))
print(dt_k_h[0],dt_k_h[-1],len(dt_k_h))
print(len(k),len(k_h))


for w in (10,30,50,70):
    corr_coeff_k=np.corrcoef(gaussian_filter1d(k,w),gaussian_filter1d(k_h,w))[1][0]
    fig,((ax1),(ax2)) = plt.subplots(2,figsize=(10,8))
    ax1.set_title('Smoothing window='+str(w)+' days')
    ax1.plot(dt_k,gaussian_filter1d(k,w),'r',label='K index Budkov')
    ax1.legend(loc='best')
    ax1.set_ylabel('K index')
    ax2.plot(dt_k_h,gaussian_filter1d(k_h,w),'g',label='K index Hurbanovo 350 nT')
    ax2.legend(loc='best')
    ax2.set_ylabel('K index')
    ax2.text(0.5,1.05, 'corr_coef = '+str(round(corr_coeff_k,4)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, c='k')
    plt.xlabel('date')
    plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/kindex_budkov_hurb'+str(w)+'.pdf', format='PDF')
    plt.show()
    plt.close()

np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k_hurb.txt', np.array([poradie,dt_k_h,k_h]).T, delimiter='\t', fmt="%s")
    
print(1)