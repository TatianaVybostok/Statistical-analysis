from datetime import datetime,timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os,glob
#from scipy.ndimage import boxcarsmooth
#from scipy.ndimage import smooth_previous
from collections import Counter
import scipy.interpolate as interp
from scipy.optimize import curve_fit
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression

def date_key(a):
    b = datetime.strptime(a, '%d.%m.%Y').date()
    return b

def smooth_previous(series, days):
    # specialni zhlazovani: pro dany den se uvazuji prumerne hodnoty teplot z predchozich "days" dni. 
    # duvod: ocekava se, ze vliv budou mit udalosti predchozi, ne budouci, proto klasicke (oboustranne)
    # zhlazovani nemusi byt reprezentativni
    new_series=np.zeros_like(series)
    for i in range(days,len(series)):
        new_series[i]=np.mean(series[i-days:i])
    return new_series 

def boxcarsmooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#mean temperature tct to dataframe
path_mt = '/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/meteodata/mean_temperature/'
mean_temp = [f for f in os.listdir(path_mt) if f.endswith('.txt')]
col_mt=['souid','date','tg','q_tg']
for i,item in enumerate(mean_temp):
        if i==0:
            mean_temp_data1=pd.read_csv(str(path_mt)+str(item), sep=",", names= col_mt, header=None)
        elif i==1:
            mean_temp_data2=pd.read_csv(str(path_mt)+str(item), sep=",", names= col_mt, header=None)
        elif i==2:
            mean_temp_data3=pd.read_csv(str(path_mt)+str(item), sep=",", names= col_mt, header=None)
        else:
            pass
#mean_temp_data1_cl=mean_temp_data1[mean_temp_data1['q_tg']!=9]
mean_temp_data1.loc[mean_temp_data1.q_tg == 9, 'tg'] = 0
#mean_temp_data2_cl=mean_temp_data2[mean_temp_data2['q_tg']!=9]
mean_temp_data2.loc[mean_temp_data2.q_tg == 9, 'tg'] = 0
#mean_temp_data3_cl=mean_temp_data3[mean_temp_data3['q_tg']!=9]
mean_temp_data3.loc[mean_temp_data3.q_tg == 9, 'tg'] = 0
mean_temp_data=pd.concat([mean_temp_data1, mean_temp_data2,mean_temp_data3],ignore_index=False).groupby(['date'],as_index=False).mean('tg')
#date and temperature to list
mt_date=list(mean_temp_data['date'])
mt_date = [int(item) for item in mt_date]
mt_tg=list(mean_temp_data['tg'])


#precipitation amount txt to dataframe
path_pa = '/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/meteodata/precipitation_amount/'
prec_amount = [f for f in os.listdir(path_pa) if f.endswith('.txt')]
col_pa=['souid','date','rr','q_rr']
for i,item in enumerate(prec_amount):
        if i==0:
            prec_amount_data1=pd.read_csv(str(path_pa)+str(item), sep=",", names= col_pa, header=None)
        elif i==1:
            prec_amount_data2=pd.read_csv(str(path_pa)+str(item), sep=",", names= col_pa, header=None)
        elif i==2:
            prec_amount_data3=pd.read_csv(str(path_pa)+str(item), sep=",", names= col_pa, header=None)
        else:
            pass
#prec_amount_data1_cl=prec_amount_data1[prec_amount_data1['q_rr']!=9]
prec_amount_data1.loc[prec_amount_data1.q_rr == 9, 'rr'] = 0
#prec_amount_data2_cl=prec_amount_data2[prec_amount_data2['q_rr']!=9]
prec_amount_data2.loc[prec_amount_data2.q_rr == 9, 'rr'] = 0
#prec_amount_data3_cl=prec_amount_data3[prec_amount_data3['q_rr']!=9]
prec_amount_data3.loc[prec_amount_data3.q_rr == 9, 'rr'] = 0
prec_amount_data=pd.concat([prec_amount_data1, prec_amount_data2,prec_amount_data3],ignore_index=False).groupby(['date'],as_index=False).mean('rr')
#date and precipitation amount to list
pa_date=list(prec_amount_data['date'])
pa_date = [int(item) for item in pa_date]
pa_rr=list(prec_amount_data['rr'])


#kindex
#HURBANOVO
filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nt/k_value/k-3.txt'

f = open(filein, 'r')
lines = f.readlines()
f.close()
x=[]
dates=[]
k=[]
for line in lines:
        p = line.split()
        x.append(float(p[0])) #poradie#
        dates.append(p[1])
        k.append(float(p[2])) #kindex                
poradie = np.array(x)
kindex = np.array(k)
kindex_date_arr=[]
for i,item in enumerate(dates):
    kindex_date_arr.append(datetime.strptime(str(item), '%d.%m.%Y'))
#datetiem to string in format YYYYmmdd    
kindex_date=[date_obj.strftime('%Y%m%d') for date_obj in kindex_date_arr]


#poruchy
file_datumy_poruch='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-1-poruchy.txt'  
f3=open(file_datumy_poruch, 'r')
lines3 = f3.readlines()
f3.close()

date_=[]
for line in lines3:
    p=line.split()
    date_.append(str(p[0]))# datum poruch

dist_dates=[]
for i in date_:            
    dist_dates.append(date_key(i))
binss=pd.date_range(dist_dates[0],dist_dates[-1],freq='d')
binss_str=binss.strftime('%Y%m%d')   

fail_date_arr=[]
for i,item in enumerate(date_):
    fail_date_arr.append(datetime.strptime(str(item), '%d.%m.%Y'))

#datetiem to string in format YYYYmmdd    
fail_date=[date_obj.strftime('%Y%m%d') for date_obj in fail_date_arr]

#for K INDEX AND METEODATA
first_ind=[]
for i,item in enumerate(kindex_date):
    if item == str(pa_date[0]):
        print(i)
        first_ind=i

last_ind=[]
for i,item in enumerate(pa_date):
    if str(item) == (kindex_date[-1]):
        last_ind=i

#OREZANIE POLI        
kindex_k=kindex[first_ind:]
mt_tg_k=mt_tg[:last_ind+1]
pa_rr_k=pa_rr[:last_ind+1]


#FOR FAILURES AND METEODATA
strt_ind=[]
end_ind=[]
for i,item in enumerate(mt_date):
    if str(item) == (fail_date[0]):
        strt_ind = i
    elif str(item) == (fail_date[-1]):
        end_ind = i
    else:
        pass

#FOR FAILURES AND KINDEX
strt_ind_k=[]
end_ind_k=[]
for i,item in enumerate(kindex_date):
    if str(item) == (fail_date[0]):
        strt_ind_k = i
    elif str(item) == (fail_date[-1]):
        end_ind_k = i
    else:
        pass

#orezanie na rovnake indexy
mt_date=mt_date[strt_ind+2:end_ind+1]
mt_tg=mt_tg[strt_ind+2:end_ind+1]

pa_date=pa_date[strt_ind+2:end_ind+1]
pa_rr=pa_rr[strt_ind+2:end_ind+1]

k_date=kindex_date[strt_ind_k+2:end_ind_k+1]
k_val=kindex[strt_ind_k+2:end_ind_k+1]


#string date to datetime in order to do ordinal dates
mt_date_dt=[]
for i,item in enumerate(mt_date):
    mt_date_dt.append(datetime.strptime(str(item), '%Y%m%d'))
pa_date_dt=[]
for i,item in enumerate(pa_date):
    pa_date_dt.append(datetime.strptime(str(item), '%Y%m%d'))
k_date_dt=[]
for i,item in enumerate(k_date):
    k_date_dt.append(datetime.strptime(str(item), '%Y%m%d'))

#BINOVANIE DAT
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

D2=mt_date_dt
Y2=mt_tg

D3=pa_date_dt
Y3=pa_rr

D4=k_date_dt
Y4=k_val


####################################################################################################################################################
sigma=20
#FITOVANIE
MT=smooth_previous(Y2,sigma)/10
DIST=boxcarsmooth(np.asarray(Y1).astype('float'),sigma)

def polynom(x, a, b, c, d ):
    return a*x*x*x + b*x*x + c*x + d
    #return d + c * x + b * np.exp(a * x)

popt, _ = curve_fit(polynom, MT[::sigma], DIST[::sigma]) #kazdy n prvok kvoli zhladzovaniu sigmaa=n aby sme dostali nezlavisle hodnoty
a,b,c,d=popt
fit_eq=('y = %.10f * x^3 + %.10f * x^2 + %.10f * x + %.10f' % (a, b, c, d))


####################################################################################################################################################
#BINOVANIE DAT
Y2=[x/10 for x in Y2]
from scipy import stats
bin_mean, bin_edges_mean, binnumber_mean =stats.binned_statistic(MT[::sigma], DIST[::sigma], 'mean', bins=100)
bin_std, bin_edges_std, binnumber_std=stats.binned_statistic(MT[::sigma], DIST[::sigma], 'std', bins=100)

print(min(Y2), min(MT[::sigma]),max(Y2), max(MT[::sigma]))


plt.errorbar(bin_edges_mean[:-1], bin_mean, bin_std,color='black', linestyle='None', marker='o', markersize=4, ecolor= 'lightgray',elinewidth=2, label='normal data')
plt.xlabel('Bins edges')
plt.ylabel('Daily anomaly rate')
x_line = np.arange(min(bin_edges_mean[:-1]), max(bin_edges_mean[:-1]), 1)
y_line = polynom(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color = 'red')
plt.plot(MT[::sigma],DIST[::sigma], markersize=2,label='real data',linestyle='None', marker="o",color='green')
plt.legend()
#plt.text(-7, 0.53, 'Smoothing widnow = ' +str(sigma), color = 'k')
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/MT_bins.pdf', format='PDF')
plt.show()
####################################################################################################################################################

print(1)