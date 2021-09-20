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

#BUDKOV
#filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex/k-3.txt'

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

#['09.07.2011', '10.08.2012', '08.07.2015']
peak_dates=[datetime(2011,7,9),datetime(2012,8,10),datetime(2015,7,8)]
peaks=[[datetime(2011,5,9),datetime(2011,9,9)],
       [datetime(2012,6,10),datetime(2012,10,10)],
       [datetime(2015,5,8),datetime(2015,9,8)]]

st_ind=[]
ed_ind=[]
mydates=[]
for j,peak in enumerate(peaks):
    mydates.append(pd.date_range(peak[0],peak[1]-timedelta(days=1),freq='d'))
    for i,item in enumerate(kindex_date_arr):
        if item ==peak[0]:
            st_ind.append(i)
        elif item==peak[1]:
            ed_ind.append(i)
    fig,(ax1) = plt.subplots(1,figsize=(8,6))
    ax1.plot(mydates[j],kindex[st_ind[j]:ed_ind[j]], 'k' ,label='k index')
    ax1.axvline(x=peak_dates[j],c='red',ls='--', lw=1)
    ax1.text(0.6, 0.8, str(peak_dates[j].date()), horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, c='red')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=30)
    ax1.legend(loc='best')
    #plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/kindex_for_special_peaks'+str(peak_dates[j].date())+'.eps', format='PDF')
    #plt.show()
    plt.close()

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


#VYKRESLENIE GRAFOV
fig, ((ax1), (ax2),(ax3)) = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False,figsize=(10,8))
ax1.plot(kindex_k, label='k-index')
ax1.legend()
ax2.plot(mt_tg_k, label='priemerne teploty')
ax2.legend()
ax3.plot(pa_rr_k, label='uhrn zrazok')
ax3.legend()
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/meteodata_vs_kindex.pdf', format="PDF")
#plt.show()
plt.close()

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


# ##ulozenie dat do txt files
dat_mt = np.array([D2, Y2])
dat_mt = dat_mt.T
np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/meteodata/mean_temperature/mean_temp_.txt', dat_mt , delimiter='\t', fmt="%s")

dat_pa = np.array([D3, Y3])
dat_pa = dat_pa.T
np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/meteodata/precipitation_amount/prec_amount_.txt', dat_pa , delimiter='\t', fmt="%s")

# for window in [10,20,30,40,50,60,70,80,90,100]:
#     mean_temp_smooth=smooth_previous(np.asarray(Y2).astype('float'),window)
#     dat_mt = np.array([D2, mean_temp_smooth])
#     dat_mt = dat_mt.T

#     prec_amount_smooth=smooth_previous(np.asarray(Y3).astype('float'),window)
#     dat_pa = np.array([D3, prec_amount_smooth])
#     dat_pa = dat_pa.T

#     np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/meteodata/mean_temp_' + str(window) + '.txt', dat_mt , delimiter='\t', fmt="%s")
#     np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/meteodata/prec_amount_' + str(window) + '.txt', dat_pa , delimiter='\t', fmt="%s")


sigma=20
#print(min(smooth_previous(Y2,sigma)/10))

corr_coeff_k=np.corrcoef(boxcarsmooth(np.asarray(Y1).astype('float'),sigma),boxcarsmooth(Y4,sigma))[1][0]
corr_coeff_mt=np.corrcoef(boxcarsmooth(np.asarray(Y1).astype('float'),sigma),smooth_previous(Y2,sigma)/10)[1][0]
corr_coeff_pa=np.corrcoef(boxcarsmooth(np.asarray(Y1).astype('float'),sigma),smooth_previous(Y3,sigma))[1][0]


fig, ((ax1),(ax2),(ax3),(ax4)) = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False,figsize=(10,8))
ax1.set_title('Smoothing window '+str(sigma)+' days')
ax1.plot(D1, boxcarsmooth(np.asarray(Y1).astype('float'),sigma),'r',label='disturbances')
ax1.legend()
ax2.plot(D4, boxcarsmooth(Y4,sigma),'k',label='kindex')
ax2.legend()
ax2.text(0.85, 0.1, 'corr_coef = '+str(round(corr_coeff_k,4)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, c='k')
ax3.plot(D2, smooth_previous(Y2,sigma)/10,label='mean temperature')
ax3.legend()
ax3.text(0.85, -1.1, 'corr_coef = '+str(round(corr_coeff_mt,4)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,c='C0')
ax4.plot(D3, smooth_previous(Y3,sigma),'g',label='precipitation amount',)
ax4.legend()
ax4.text(0.85, -1.8, 'corr_coef = '+str(round(corr_coeff_pa,4)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, c='g')
plt.xlabel('dates')
plt.xticks(rotation=45, ha='right')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/prec_amount_mean_temp_dist.pdf', format='PDF')
#HURBANOVO
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/prec_amount_mean_temp_dist.pdf', format='PDF')
#plt.show()
plt.close()


####################################################################################################################################################
#FITOVANIE
MT=smooth_previous(Y2,sigma)/10
DIST=boxcarsmooth(np.asarray(Y1).astype('float'),sigma)

def polynom(x, a, b, c, d ):
    return a*x*x*x + b*x*x + c*x + d
    #return d + c * x + b * np.exp(a * x)

popt, _ = curve_fit(polynom, MT[::sigma], DIST[::sigma]) #kazdy n prvok kvoli zhladzovaniu sigmaa=n aby sme dostali nezlavisle hodnoty
a,b,c,d=popt
fit_eq=('y = %.10f * x^3 + %.10f * x^2 + %.10f * x + %.10f' % (a, b, c, d))
print(fit_eq)

#a = 0.00001 
#b = 0.00047
#c = 0.00508 
#d = 0.07031

plt.plot(MT[::sigma],DIST[::sigma], markersize=1,label='sigma='+str(sigma),linestyle='None', marker="o")
plt.xlabel('mean temperature [$^\circ$C]')
plt.ylabel('Daily anomaly rate (smoothed & binned)')
plt.legend(loc='best')
x_line = np.arange(min(MT[::sigma]), max(MT[::sigma]), 1)
# calculate the output for the range
y_line = polynom(x_line, a, b, c, d)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color = 'red')
#plt.text(0, 0.1, fit_eq, color = 'red')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/MT_DIST_function_boxcarsmooth_test.pdf', format='PDF')
#HURBANOVO
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/MT_DIST_function_boxcarsmooth_test.pdf', format='PDF')
plt.show()
plt.close()

#rezidua a test
DIST_COMPT=[]
for i in MT[::sigma]:
    DIST_COMPT.append(polynom(i,a,b,c,d))

rezid=DIST[::sigma]-DIST_COMPT
plt.plot(MT[::sigma],rezid,'o',markersize=1)
plt.xlabel('mean temperature [$^\circ$C]')
plt.ylabel('Residual of disturbances')
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/MT_DIST_rezid_boxcarsmooth.pdf', format='PDF')
#plt.show()
plt.close()

plt.hist(rezid, density=True, bins=10)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Residual disturbances')
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/MT_DIST_rezid_HIST_boxcarsmooth.pdf', format='PDF')
#plt.show()
plt.close()

##########################################################################################################################################################################
# MODELOVE PORUCHY

#pocitam model pre nezhladene priemerne teploty
model_dist=[]  
for i in MT:
    #model_dist.append(round(polynom(i,a,b,c,d)))
    model_dist.append(polynom(i,a,b,c,d))
np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/model_dist'+str(sigma)+'.txt', model_dist,fmt='%1.4f')

#zaporne hodnoty prahujem nulou
new_model_dist = []
for ind, item in enumerate(model_dist):
    if item <= 0:
        new_model_dist.append(0)
    else:
        new_model_dist.append(item)

#modelove poruchy bez teplotnej zavislosti
#rezid_dist = boxcarsmooth(np.asarray(Y1).astype('float'),sigma) - new_model_dist 
#new_model_dist=[round(num) for num in new_model_dist]
rezid_dist = Y1 - new_model_dist

print(max(Y1),max(new_model_dist),min(Y1),min(new_model_dist))

# #len dalsie prahovanie nulou, nasobenie 100 aby sme mali cele cisla
new_rezid_dist = []
for ind, item in enumerate(rezid_dist):
    if item <= 0:
        new_rezid_dist.append(0)
    else:
        new_rezid_dist.append(round(item))


fig, ((ax1),(ax2),(ax3), (ax4)) = plt.subplots(4,figsize=(10,8))
ax1.plot(Y1, color='darkblue', label='original dist')
ax1.legend(loc='best')
ax2.plot(DIST, 'g', label='boxcarsmoothed dist')
ax2.legend(loc='best')
ax3.plot(new_model_dist,'r', label='modeled dist')
ax3.legend(loc='best')
ax4.plot(new_rezid_dist, color= 'purple', label='rezid dist')
ax4.legend(loc='best')
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/boxcarsmooth_rezid.pdf', format='PDF')
#plt.show()
plt.close()

import time
new_D1=[]
for i in D1:
    new_D1.append(i.strftime("%d.%m.%Y"))

model_dist_dates=[]
for ind, item in enumerate(new_rezid_dist):
    if item > 0:
        model_dist_dates.extend( [new_D1[ind]] * int(item))


#np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-1-poruchy_rezid_boxcarsmooth.txt', model_dist_dates,fmt='%s')
np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-1-poruchy_rezid_test.txt', model_dist_dates,fmt='%s')

##########################################################################################################################################################################
#REZIDUA
model_dist_dt=[]
for i, item in enumerate(model_dist_dates):
    model_dist_dt.append((datetime.strptime(str(item), '%d.%m.%Y')))

#BINOVANIE DAT
model_ord_dist=[]
for i in range(1,len(model_dist_dt)):
  model_ord_dist.append(datetime.toordinal(model_dist_dt[i]))

model_X=np.histogram(model_ord_dist, ord_bins)

model_dat=(model_X[1])  #[:-1]
model_peaks=model_X[0]
model_Y=model_peaks

model_D=[]
for i in range(1,len(model_dat)):
  model_D.append(datetime.fromordinal(model_dat[i]))

rezid_dist_divided = [x  for x in new_rezid_dist]
rezid_model_seps =  boxcarsmooth(np.asarray(Y1).astype('float'),sigma) - new_model_dist  #rezid_dist_divided - boxcarsmooth(np.asarray(Y1).astype('float'),sigma)

#fig, ((ax1),(ax2),(ax3),(ax4),(ax5),(ax6),(ax7),(ax8)) = plt.subplots(nrows=8, ncols=1, sharex=True, sharey=False,figsize=(12,10))
fig, ((ax1),(ax2),(ax3),(ax4),(ax5),(ax6),(ax7),(ax8)) = plt.subplots(nrows=8, ncols=1, sharex=True, sharey=False,figsize=(12,10))

ax1.set_title('Smoothing window '+str(sigma)+' days')
ax1.plot(D1,Y1,'r',label='SEPS dist')
ax1.legend()
ax2.plot(D1, boxcarsmooth(np.asarray(Y1).astype('float'),sigma),'r',label='smoothed SEPS dist')
ax2.legend()
ax3.plot(model_D, boxcarsmooth(np.asarray(new_model_dist).astype('float'),sigma),'orange',label='modeled dist')
ax3.legend()
#ax4.plot(model_D,rezid_model_seps,'k',label='reziduum modeled & smoothed SEPS dist')
#ax4.legend()
#ax5.plot(model_D,boxcarsmooth(np.asarray(rezid_model_seps).astype('float'),sigma),'k',label='smoothed reziduum modeled & SEPS dist')
#ax5.legend()
ax4.plot(model_D,rezid_dist_divided,label='reziduum')
ax4.legend()
ax5.plot(model_D, boxcarsmooth(np.asarray(rezid_dist_divided).astype('float'),sigma),label='smoothed reziduum')
ax5.legend()



#ax7.xaxis.set_tick_params(labelbottom=True, size=2)
ax6.plot(D4, boxcarsmooth(Y4,sigma),'k',label='kindex')
ax6.legend()
#ax3.xaxis.set_tick_params(labelbottom=True, size=2)
#ax4.xaxis.set_tick_params(labelbottom=True, size=2)
ax7.plot(D2, MT,'m',label='mean temperature')
ax7.legend()
ax8.plot(D3, boxcarsmooth(Y3,sigma),'g',label='precipitation amount')
ax8.legend()
plt.xlabel('dates')
#plt.xticks(rotation=45, ha='right')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/mean_temp_modeled_dist_boxcarsmooth_FINAL_SEPS.pdf', format='PDF')
#HURBANOVO
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/mean_temp_modeled_dist_boxcarsmooth_FINAL_SEPS.pdf', format='PDF')
plt.show()
plt.close()

with open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/Michal_data.txt','w') as f:
    lis=[np.array(D1),np.array(Y1),np.array(MT)]
    for x in zip(*lis):
        f.write("{0}\t{1}\t{2}\n".format(*x))
#MODELED
sigmas=range(1,200)
Corr_mt=[]
Corr_pa=[]
Corr_k=[]
for isigma in sigmas:
  Corr_mt.append((np.corrcoef(boxcarsmooth(np.asarray(model_Y).astype('float'),isigma),boxcarsmooth(np.asarray(Y2).astype('float'),isigma)))[0,1])
  Corr_pa.append((np.corrcoef(boxcarsmooth(np.asarray(model_Y).astype('float'),isigma),boxcarsmooth(np.asarray(Y3).astype('float'),isigma)))[0,1])
  Corr_k.append((np.corrcoef(boxcarsmooth(np.asarray(model_Y).astype('float'),isigma),boxcarsmooth(np.asarray(Y4).astype('float'),isigma)))[0,1])

fig,((ax1),(ax2),(ax3))=plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False,figsize=(10,8))
ax1.plot(sigmas, Corr_k, 'k', label='k index')
ax1.legend()
ax1.set_xlabel('Smoothing window [days]')
ax1.set_ylabel('Correlation coefficient')
ax1.xaxis.set_tick_params(labelbottom=True, size=10)
ax2.plot(sigmas, Corr_mt, label='mean temperature')
ax2.legend()
ax2.set_xlabel('Smoothing window [days]')
ax2.set_ylabel('Correlation coefficient')
ax2.xaxis.set_tick_params(labelbottom=True, size=10)
ax3.plot(sigmas, Corr_pa, 'g', label='precipitation amount')
ax3.legend()
ax3.set_xlabel('Smoothing window [days]')
ax3.set_ylabel('Correlation coefficient')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/korelace_prec_amount_mean_temp_modeled_dist_boxcarsmooth_FINAL.pdf', format='PDF')
#HURBANOVO
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/korelace_prec_amount_mean_temp_modeled_dist_boxcarsmooth_FINAL.pdf', format='PDF')
plt.show()
plt.close()

#ORIGINAL
sigmas=range(1,200)
Corr_mt=[]
Corr_pa=[]
Corr_k=[]
for isigma in sigmas:
  Corr_mt.append((np.corrcoef(boxcarsmooth(np.asarray(Y1).astype('float'),isigma),boxcarsmooth(np.asarray(Y2).astype('float'),isigma)))[0,1])
  Corr_pa.append((np.corrcoef(boxcarsmooth(np.asarray(Y1).astype('float'),isigma),boxcarsmooth(np.asarray(Y3).astype('float'),isigma)))[0,1])
  Corr_k.append((np.corrcoef(boxcarsmooth(np.asarray(Y1).astype('float'),isigma),boxcarsmooth(np.asarray(Y4).astype('float'),isigma)))[0,1])

fig,((ax1),(ax2),(ax3))=plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False,figsize=(10,8))
ax1.plot(sigmas, Corr_k, 'k', label='k index')
ax1.legend()
ax1.set_xlabel('Smoothing window [days]')
ax1.set_ylabel('Correlation coefficient')
ax1.xaxis.set_tick_params(labelbottom=True, size=10)
ax2.plot(sigmas, Corr_mt, label='mean temperature')
ax2.legend()
ax2.set_xlabel('Smoothing window [days]')
ax2.set_ylabel('Correlation coefficient')
ax2.xaxis.set_tick_params(labelbottom=True, size=10)
ax3.plot(sigmas, Corr_pa, 'g', label='precipitation amount')
ax3.legend()
ax3.set_xlabel('Smoothing window [days]')
ax3.set_ylabel('Correlation coefficient')
#BUDKOV
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/korelace_prec_amount_mean_temp_modeled_dist_boxcarsmooth_FINAL.pdf', format='PDF')
#HURBANOVO
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/orig_korelace_prec_amount_mean_temp_modeled_dist_boxcarsmooth_FINAL.pdf', format='PDF')
plt.show()
plt.close()

#import code
#code.interact(local=locals())
print('finished')


    




            
