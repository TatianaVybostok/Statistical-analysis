import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import InterpolatedUnivariateSpline, Rbf
from scipy import interpolate
import random
from scipy import stats

#window=10
#distributor=1

#poradie
f1 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/poradie.txt', 'r')
lines1 = f1.readlines()
f1.close()

x=[]
b=[]
c=[]
d=[]

for line in lines1:
        p = line.split()
        x.append(float(p[0]))

xv = np.array(x)

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

#maxima
b = []
f2 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/sortedmax.txt', 'r')
lines2 = f2.readlines()
f2.close()

for line in lines2:
        p = line.split()
        b.append(float(p[0]))
bv = np.array(b)

#minima        
c = []
f3 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/sortedmin.txt', 'r')
lines3 = f3.readlines()
f3.close()

for line in lines3:
        p = line.split()
        c.append(float(p[0]))
cv = np.array(c)


# nahodne intervaly
d=[]
for i in range(len(cv)*3):
  d.append(random.randint(min(xv),max(xv)))
dv = np.array(d)


wc=np.where((cv<(max(xv)-window/2)) & (cv>(min(xv)+window/2)))
cv=np.take(cv,wc)
#print(c,cv)

#w=np.where((bv<(max(xv)-window/2)) & (bv>(min(xv)+window/2)))
wb=np.where((bv<(max(xv)-window)) & (bv>(min(xv)+1)))
bv=np.take(bv,wb)
#print(b,bv)

cv=cv[0]
bv=bv[0]


wd=np.where((dv<(max(xv)-window)) & (dv>(min(xv)+1)))
dv=np.take(dv,wd)
dv=dv[0]

#print (bv)
#print (cv)
#volim ten rozsah ktory je mensi
if len(bv)>len(cv):
   lenght=cv
else:
   lenght=bv
lenght=np.array(lenght)

    
h1=0
j1=0
h2=0
j2=0
h3=0
j3=0


bvv=bv.tolist()
cvv=cv.tolist()
dvv=dv.tolist()
             

# definice clenu pro risk
R_a=0.0
R_b=0.0
R_c=0.0
R_d=0.0

#print bvv[0]
#-----------------------------------
#co se deje kolem maxima

idaysmax=[]
pdaysmax=[]
for iday in range(-window, window):
  idaysmax.append(iday)
  temp=0

  for i in range(len(lenght)):
    g2=int(bvv[i])+iday
    l2=int(bvv[i])+iday+1
    f2=[]
    for ii in range(g2,l2):
            f2.append(ii)
    tt2=np.take(av,f2,mode='raise')
    temp=temp+sum(int(i) for i in tt2)
     
  pdaysmax.append(temp)

idaysmin=[]
pdaysmin=[]
for iday in range(-window, window):
  idaysmin.append(iday)
  temp=0

  for i in range(len(lenght)):
    g2=int(cvv[i])+iday
    l2=int(cvv[i])+iday+1
    f2=[]
    for ii in range(g2,l2):
            f2.append(ii)
    tt2=np.take(av,f2,mode='raise')
    temp=temp+sum(int(i) for i in tt2)
     
  pdaysmin.append(temp)
  
mean_max=np.mean(pdaysmax)
#mean_max=np.median(pdaysmax)
mean_min=np.mean(pdaysmin)
std_min=np.std(pdaysmin)
  
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
#plt.plot(idays, pdays)
plt.bar(idaysmax, pdaysmax, align='center', width=1)
plt.xlabel ('Days from k-index local maximum')
plt.ylabel ('Number of failures per day')

ax.axhline(y=mean_max, color='black', linestyle='-')
ax.axhline(y=mean_max+std_min, color='black', linestyle=':')
ax.axhline(y=mean_max+2*std_min, color='black', linestyle=':')
ax.axhline(y=mean_max+3*std_min, color='black', linestyle=':')
ax.axhline(y=mean_max-std_min, color='black', linestyle=':')
ax.axhline(y=mean_max-2*std_min, color='black', linestyle=':')
ax.axhline(y=mean_max-3*std_min, color='black', linestyle=':')
plt.ylim(bottom=0)
#plt.legend([ 'smoothed', 'original'])
#plt.title('smoothed graph of k index') 
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/maximum-'+str(distributor)+'_'+str(window)+'.pdf', format="PDF")
#plt.show()
plt.close()

fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
#plt.plot(idays, pdays)
plt.bar(idaysmin, pdaysmin, align='center', width=1)
plt.xlabel ('Days from k-index local minumum')
plt.ylabel ('Number of failures per day')
ax.axhline(y=mean_min, color='black', linestyle='-')
ax.axhline(y=mean_min+std_min, color='black', linestyle=':')
ax.axhline(y=mean_min+2*std_min, color='black', linestyle=':')
ax.axhline(y=mean_min+3*std_min, color='black', linestyle=':')
ax.axhline(y=mean_min-std_min, color='black', linestyle=':')
ax.axhline(y=mean_min-2*std_min, color='black', linestyle=':')
ax.axhline(y=mean_min-3*std_min, color='black', linestyle=':')
plt.ylim(bottom=0)
#plt.legend([ 'smoothed', 'original'])
#plt.title('smoothed graph of k index') 
plt.savefig('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/plots/minimum-'+str(distributor)+'_'+str(window)+'.pdf', format="PDF")
#plt.show()
plt.close()
 
#stop 
#-----------------------------------


#maximum
for i in range(len(lenght)):#ak chcem jednodny rozsah nahradim bv/cv lenght
        #stop
        #print i
        #g2=int(bvv[i])-window/2
        #l2=int(bvv[i])+window/2
        g2=int(bvv[i])#-1
        l2=int(bvv[i])+window
        f2=[]
        for ii in range(g2,l2):
                f2.append(ii)
        t2 =np.take(yv,f2,mode='raise')
        #print sum(t2)/float(len(t2))
        h2=h2+sum(t2)/float(len(t2))
        tt2=np.take(av,f2,mode='raise')
        #print tt2
        #print sum(int(i) for i in tt2)
        j2=j2+sum(int(i) for i in tt2)
        #print j2
        w=np.where(tt2==0)
        R_b=R_b+len(w[0])
        R_a=R_a+(len(tt2)-len(w[0]))
#minimum   
for i in range(len(lenght)):
    #print i
    g1=int(cvv[i])-window/2
    l1=int(cvv[i])+window/2
    f1=[]
    for ii in range(int(g1),int(l1)):
            f1.append(ii)
    t1 =np.take(yv,f1,mode='raise')
    #print sum(t1)/float(len(t1))
    h1=h1+sum(t1)/float(len(t1))
    tt1=np.take(av,f1,mode='raise')
    #print tt1
    #print sum(int(i) for i in tt1)
    j1=j1+sum(int(i) for i in tt1)
    #print j1
    w=np.where(tt1==0)
    R_d=R_d+len(w[0])
    R_c=R_c+(len(tt1)-len(w[0]))
    
#random       
for i in range(len(lenght)):
   #print i
   g3=int(dvv[i]-window/2)
   l3=int(dvv[i]+window/2)
   f3=[]
   for ii in range(g3,l3):
           f3.append(ii)
   #print f3
   t3 =np.take(yv,f3,mode='raise')
   #print sum(t3)/float(len(t3))
   h3=h3+sum(t3)/float(len(t3))
   tt3=np.take(av,f3,mode='raise')
   #print tt3
   #print sum(int(i) for i in tt3)
   j3=j3+sum(int(i) for i in tt3)
   #print j3
    
j1=round(j1/100)
j2=round(j2/100)
j3=round(j3/100)

print ('Vysledky')
print ('Pocet poruch v minimu:'+str(j1)) #minimum
print ('Pocet poruch v maximu:'+str(j2)) #maximum
print ('Pocet poruch v random:'+str(j3)) 

svyznam1=stats.binom_test(j1, n=j1+j2, p=0.5)
svyznam2=stats.binom_test(j2, n=j3+j2, p=0.5)
svyznam13=stats.binom_test(j1, n=j1+j3, p=0.5)
print ('p-value H-L: '+str('{:.2E}'.format(round(svyznam1,200))))
print ('p-value R-L: '+str('{:.2E}'.format(round(svyznam2,200))))
print ('p-value R-H: '+str('{:.2E}'.format(round(svyznam13,200))))
#print round(svyznam2,5)
#print round(svyznam13,5)

#if j1>0:
  #relativneriziko=j2/float(j1)
#else:
  #relativneriziko=np.nan
  
relativneriziko=(np.float(R_a)/(R_a+R_b))/(np.float(R_c)/(R_c+R_d))
print ('r='+str(round(relativneriziko,5)))

#flog= open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/tables/table-'+str(window)+'.tex',"w+")
#flog.write('Dataset & Dataset ID & Intervals & $n_H$ & $n_R$ & $n_L$ & $p_{HL}$ & $p_{RL}$ & $p_{RH}$ & $R$ \\\\ \n')
#flog.write('\\hline \n')

flog.write(str(window)+ ' & ' +str(len(lenght))+' & ' +str(j2)+' & '+str(j1)+' & '+str('{:.2E}'.format(round(svyznam1,200)))+' & '+str(round(relativneriziko,5))+ '\\\\' +'\n')
#flog.write('D'+str(distributor)+ ' & ' +str(len(lenght))+' & '+str(j2)+' & '+str(j3)+' & '+str(j1)+' & '+str(round(svyznam1,5))+' & '+str(round(svyznam2,5))+' & '+str(round(svyznam13,5))+ str(round(relativneriziko),5) +'\\\\\n')

#frisk.write('D'+str(distributor)+' & '+str(int(R_a))+' & '+str(int(R_b))+' & '+str(int(R_c))+' & '+str(int(R_d))+' & '+str(round(relativneriziko,5))+'\\\\\n')
#print(1)






        
