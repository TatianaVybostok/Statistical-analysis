import matplotlib.pyplot as plt
import numpy as np
import csv
#from scipy.interpolate import InterpolatedUnivariateSpline, Rbf, make_interp_spline
#from scipy import interpolate
import random
from scipy import stats
from dateutil import parser
import ast

#window=70
#for window in [10, 30, 50, 70]:
#chceme_spolecne=0

x=[]
dates=[]
y=[]
b=[]
c=[]


#hurbanovo
#filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k-' + str(window)+'.txt'

#budkov
#filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/filtered/filtered_k-' + str(window)+'.txt'

f = open(filein, 'r')
lines = f.readlines()
f.close()

for line in lines:
        p = line.split()
        x.append(float(p[0])) #poradie
        dates.append(p[1])
        y.append(float(p[2])) #kindex
        b.append(float(p[3])) #prvaderiv
        c.append(float(p[4])) #druhaderiv
                
xv = np.array(x)
yv = np.array(y)
bv = np.array(b)  
cv = np.array(c)

#budkov
# last_ind=5477
# first_ind=1461
# xv=range(1,last_ind-first_ind+1)
# yv=yv[first_ind:last_ind]
# cv=cv[first_ind:last_ind]
# bv=bv[first_ind:last_ind]


#sem urezani na spolecne intervaly

epsilon = 10E-5
extremy=[]
minmax=[]

#hladam extrem k indexu, teda hodnoty prvych derivaci blizkych nule
bvv=abs(bv)
print((len(xv)))

#for i in range(len(bvv)):
#        if bvv[i]<=epsilon:
#                extremy.append(bvv[i])

# hledani extremu: prochazi se pole derivaci a hleda se, kdy derivace prochazi nulou. Kvuli vzorkovani
# tam nulove body nejsou. Ale pokud se hodnota meni ze zaporne na kladnou nebo obracene, musi mezi
# nimi nulou prochazet. Soucasne se ze znamenek 1. derivace da urcit, zda jde o maximum nebo o minimum
# Pokud se derivace meni z kladne na zapornou, musi jit o maximum, pokud se meni ze zaporne na kladnou,
# musi jit o minimum. Do pole extremy se tedy ulozi index (souradnice) extremu, do pole minmax pak +1 pro
# maximum a -1 pro minimum.

for i in (list(range(len(bv)-1-2*window))):
  j=i+window
  if (bv[j]<0) and (bv[j+1]>0):
      extremy.append(j)
      minmax.append(-1)
  if (bv[j]>0) and (bv[j+1]<0):
      extremy.append(j)
      minmax.append(+1)

print((len(extremy)))
# rozdelit na maxima a minima a odstranit prekryvajici se.
  
w=np.where(np.array(minmax)==1)
w=w[0]
maxima=list(np.take(extremy,w))
w=np.where(np.array(minmax)==-1)
w=w[0]
minima=list(np.take(extremy,w))

# pri odstranovani prekryvajicich se se vzdy vezme to vyssi maximum nebo nizsi minimum
dt=np.roll(maxima,-1)-maxima
dt1=dt.tolist()
del dt1[-1]
torem=(list(np.where(np.array(dt1)<window)))[0]  # list of indices to be removed
while len(torem)>0:
  # vzdy odstranime jen prvni vyskyt blizkych bodu, pote znovu spocitame vzdalenost a opakujeme, dokud nejsou
  # vsechny vzdalenosti dostatecne velke
  i=0
  index=torem[i] #index extremu ktory musim zmazat
  tindex=maxima[index] 
  tindex1=maxima[index+1]
  if yv[tindex]>yv[tindex1]:
    del maxima[index+1]
    print(('Removing '+str(index+1)))
  else:
    del maxima[index]
    print(('Removing '+str(index)))
  dt=np.roll(maxima,-1)-maxima
  dt1=dt.tolist()	
  del dt1[-1]
  torem=(list(np.where(np.array(dt1)<window)))[0]  # list of indices to be removed


dt=np.roll(minima,-1)-minima
dt1=dt.tolist()
del dt1[-1]
torem=(list(np.where(np.array(dt1)<window)))[0]  # list of indices to be removed
while len(torem)>0:
  # vzdy odstranime jen prvni vyskyt blizkych bodu, pote znovu spocitame vzdalenost a opakujeme, dokud nejsou
  # vsechny vzdalenosti dostatecne velke
  i=0
  index=torem[i]
  tindex=minima[index]
  tindex1=minima[index+1]
  if yv[tindex]<yv[tindex1]:
    del minima[index+1]
    print(('Removing '+str(index+1)))
  else:
    del minima[index]
    print(('Removing '+str(index)))
  dt=np.roll(minima,-1)-minima
  dt1=dt.tolist()	
  del dt1[-1]
  torem=(list(np.where(np.array(dt1)<window)))[0]  # list of indices to be removed

# znovu postavit pole minim a maxim
extremy=minima+maxima
indsort=np.argsort(extremy)
minmax=list(np.take(list([-1]*len(minima))+list([1]*len(maxima)),indsort))
extremy=list(np.take(extremy, indsort))

print((len(extremy)))


# a ted to projit kompletne 
dt=np.roll(extremy,-1)-extremy
dt1=dt.tolist()
del dt1[-1]
torem=(list(np.where(np.array(dt1)<window)))[0]  # list of indices to be removed
# jedeme po jednom
while len(torem)>0:
  # vzdy odstranime jen prvni vyskyt blizkych bodu, pote znovu spocitame vzdalenost a opakujeme, dokud nejsou
  # vsechny vzdalenosti dostatecne velke
  i=0
  index=torem[i]
  tindex=extremy[index]
  tindex1=extremy[index+1]
  #if (minmax[index]==-1) and (minmax[index+1]==+1):
    #del minmax[index]
    #del extremy[index]
    #print 'Removing '+str(index)
  #elif (minmax[index]==+1) and (minmax[index+1]==-1):
    #del minmax[index+1]
    #del extremy[index+1]
    #print 'Removing '+str(index+1)
  #else:
    #print 'Problem!'
  del extremy[index+1]
  del minmax[index+1]
  print(('Removing '+str(index+1)))
  dt=np.roll(extremy,-1)-extremy
  dt1=dt.tolist()	
  del dt1[-1]
  torem=(list(np.where(np.array(dt1)<window)))[0]  # list of indices to be removed
    
print((len(extremy)))
#if window==200:https://www.application.mps.mpg.de/public/phd.php?mode=1&step=4&app_type=
  #stop

# pozice maxim a minim v poli extremy
maxpos=(list(np.where(np.array(minmax)==1)))[0]
minpos=(list(np.where(np.array(minmax)==-1)))[0]


#hladam nablizsie minimum=closestmin
closestmin=[]
disttomin=[]
for i in range(len(maxpos)):
  m=len(bv) #prva derivace
  mp=len(minpos)
  for j in range(len(minpos)):
    dist=abs(extremy[maxpos[i]]-extremy[minpos[j]])
    if  dist< m:
      m=dist
      mp=minpos[j]
  closestmin.append(mp)
  disttomin.append(m)


# odstranit duplicity
sclosestmin=[]
for i in range(len(closestmin)):
  if not closestmin[i] in sclosestmin:
    sclosestmin.append(closestmin[i])


#hladam blizke maxima ku minimam
nfinal=len(sclosestmin)
finalmax=[]
finalmin=[]

#uprava na rovnaku dlzku extremov
for i in range(nfinal):
  # pro kazdy unikatni index hledej v poli odpovidajicich maxim hodnotu, kde je k-index maximalni
  w=maxpos[np.where(closestmin==sclosestmin[i])]
  m=-20
  mp=len(w)+1
  for j in range(len(w)):
    kindex=yv[extremy[w[j]]]
    if kindex>m:
      m=kindex
      mp=w[j]
  
  finalmin.append(extremy[sclosestmin[i]])
  finalmax.append(extremy[mp])

print ('Dobehlo')

finalextremy=finalmin+finalmax
finalextremy.sort()

np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/extremyy.txt.',finalextremy, fmt='%i')
np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/minimum.txt',finalmin, fmt='%i')
np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/maximum.txt',finalmax, fmt='%i')


fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot( xv, yv)
plt.xlabel ('Day number',size=14)
plt.ylabel ('K index',size=14)
plt.xticks(size = 12)
plt.yticks(size = 12)
for i in range(len(finalmax)):
  x0=finalmax[i]-window/2
  #x0=finalmax[i]
  rect = plt.Rectangle((x0, min(yv)), window, (max(yv)-min(yv)), color='b', alpha=0.3)
  ax.add_patch(rect)
for i in range(len(finalmin)):
  x0=finalmin[i]-window/2
  rect = plt.Rectangle((x0, min(yv)), window, (max(yv)-min(yv)), color='r', alpha=0.3)
  ax.add_patch(rect)

if chceme_spolecne==1:
# prekreslime obdelnik, kde je spolecny interval
  x0=dates.index(spolecne_t0)
  dx=dates.index(spolecne_t1)-dates.index(spolecne_t0)
  rect = plt.Rectangle((x0, min(yv)), dx, (max(yv)-min(yv)), color='k', alpha=0.3)
  ax.add_patch(rect)
  
#plt.legend([ 'smoothed', 'original'])
#plt.title('smoothed graph of k index') 
#budkov
#plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/plot-'+str(window)+'.pdf', format="PDF")
#hurbanovo
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/plots/plot-'+str(window)+'_hurbanovo.pdf', format="PDF")
#plt.show()
plt.close()
