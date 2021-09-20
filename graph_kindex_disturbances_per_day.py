from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def parsefile(filename):
  fhandle=open(filename,'r')
  lines=fhandle.readlines()
  fhandle.close()
  retarr=[]
  for line in lines:
    p=line.split()
    retarr.append((p[0]))
  return retarr

#budkov
#filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex/k_der.txt'
#hurbanovo
filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k_der.txt'

f = open(filein, 'r')
lines = f.readlines()
f.close()

x=[]
dates=[]
k=[]

for line in lines:
        p = line.split()
        x.append(float(p[0])) #poradie
        dates.append(p[1])
        k.append(float(p[2])) #kindex                
poradie = np.array(x)
kindex = np.array(k)
res = [idx for idx, val in enumerate(kindex) if val > 9]
print(kindex[res] )

file_datumy_poruch='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-1-poruchy.txt'  
f3=open(file_datumy_poruch, 'r')
lines3 = f3.readlines()
f3.close()

y2=[]
for line in lines3:
    p=line.split()
    y2.append(str(p[0]))# datum poruch

def date_key(a):
    b = datetime.strptime(a, '%d.%m.%Y').date()
    return b

dist_dates=[]
kindex_dates=[]
for i in y2:            
    dist_dates.append(date_key(i))

for i in dates:            
    kindex_dates.append(date_key(i))


start=dates.index(y2[0])
end=dates.index(y2[-1])

binss=pd.date_range(dist_dates[0],dist_dates[-1],freq='d')
binss_str=binss.strftime("%Y-%m-%d")

from matplotlib.ticker import MaxNLocator


fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,figsize=(12,10))
ax1.plot(kindex_dates[start:end+1],kindex[start:end+1],label='K-index')
ax1.set_xticks(np.arange(kindex_dates[start], kindex_dates[end+1], 200))
ax1.tick_params(labelrotation=45)
ax1.legend(loc='upper right')
ax1.set_ylabel('K-index')
ax2.hist(dist_dates,bins=binss,label='Dates of disturbances',color='r')
#ax2.grid()
ax2.legend(loc='upper right')
plt.xlabel('dates')
ax2.set_xticks(np.arange(dist_dates[0], dist_dates[-1], 200))
ax2.set_ylabel('Anomaly rate per day')
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(rotation=45, ha='right')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/histogram.pdf', format="PDF")
#HURBANOVO
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/histogram.pdf', format='PDF')
plt.show()

dates_kindex=np.array((dates[start:end+1]))
k_index=np.array(kindex[start:end+1])
dates_disturbance=np.array(y2)
bins_for_hist=np.array(binss_str)

#np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hist.txt', list(zip(dates_kindex,k_index)), delimiter='\t', fmt="%s")
#np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/dist_hist.txt', list(zip(dates_disturbance)), delimiter='\t', fmt="%s")
#np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/bins_hist.txt', list(zip(bins_for_hist)), delimiter='\t', fmt="%s")  

ord_bins=[]
for i in range(1,len(binss)):
  ord_bins.append(datetime.toordinal(binss[i]))

ord_dist=[]
for i in range(1,len(dist_dates)):
  ord_dist.append(datetime.toordinal(dist_dates[i]))

X=np.histogram(ord_dist, ord_bins)

dat=(X[1])  #[:-1]
peaks=X[0]

D=[]
for i in range(1,len(dat)):
  D.append(datetime.fromordinal(dat[i]))

D1=D
Y1=peaks

D2=kindex_dates[start+2:end+1]
Y2=kindex[start+2:end+1]

from scipy.ndimage import gaussian_filter1d
from scipy.stats.stats import pearsonr 

sigma=3.0


corr_coef=pearsonr(gaussian_filter1d(Y2,sigma),gaussian_filter1d(np.asarray(Y1).astype('float'),sigma))[0]

fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,figsize=(10,8))
ax1.plot(D2, gaussian_filter1d(Y2,sigma),label='K-index')
ax1.set_title('Smoothing window '+str(sigma)+' days')
ax1.legend()
ax2.plot(D1, gaussian_filter1d(np.asarray(Y1).astype('float'),sigma),'r',label='disturbances')
ax2.legend()
plt.xlabel('dates')
plt.xticks(rotation=45, ha='right')
ax2.text(0.8, 0.8, 'corr_coef = '+str(round(corr_coef,4)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, c='black')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/kindex_poruchy.pdf', format='PDF')
#HURBANOVO
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/kindex_poruchy.pdf', format='PDF')
plt.show()
plt.close()


sigmas=range(1,200)
C=[]
for isigma in sigmas:
  C.append((np.corrcoef(gaussian_filter1d(np.asarray(Y1).astype('float'),isigma),gaussian_filter1d(np.asarray(Y2).astype('float'),isigma)))[0,1])

plt.plot(sigmas, C)
#plt.title('Correlation')
plt.xlabel('Smoothing window [days]')
plt.ylabel('Correlation coefficient')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/korelace_kindex_poruchy.pdf', format='PDF')
#HURBANOVO
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/korelace_kindex_poruchy.pdf', format='PDF')
plt.show()
plt.close()

import code
code.interact(local=locals())

#np.savetxt('kindex_hist.txt', list(zip(dates_kindex,k_index)), delimiter='\t', fmt="%s")
#np.savetxt('dist_hist.txt', list(zip(dates_disturbance)), delimiter='\t', fmt="%s")
#np.savetxt('bins_hist.txt', list(zip(bins_for_hist)), delimiter='\t', fmt="%s")  

print(1)