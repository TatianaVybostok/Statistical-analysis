import numpy as np
from datetime import datetime
from dateutil import parser
import ast

# window=10
# distributor=1
# chceme_spolecne=0
# filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k-' + str(window)+'.txt'
# file_datumy='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-'+str(distributor)+'-datumy.txt'
# file_datumy_poruch='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-'+str(distributor)+'-poruchy_rezid_test.txt'

f1 = open(filein, 'r')
lines1 = f1.readlines()
f1.close()

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
y1 = []
y2 = []
z1 = []


for line in lines1:
    p=line.split()
    x1.append(float(p[0])) # poradie
    x2.append(p[1]) #datum k indexu
    x3.append(float(p[2])) #hodnota k indexu
    x4.append(float(p[3]))#prva deriv
    x5.append(float(p[4]))# druha deriv
    
f2=open(file_datumy, 'r')
lines2 = f2.readlines()
f2.close()

for line in lines2:
    p=line.split()
    y1.append((p[0]))# datum
    
f3=open(file_datumy_poruch, 'r')
lines3 = f3.readlines()
f3.close()

for line in lines3:
    p=line.split()
    #print p[0]
    y2.append((p[0]))# datum poruch
print('poruchyyyyyyyyyy'+str(y2))
#ma tu byt toto max a min???
    
f4=open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/maximum.txt', 'r')
lines4 = f4.readlines()
f4.close()
y4=[]
for line in lines4:
    p=line.split()
    y4.append((p[0]))# maximum indexy

f5=open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/minimum.txt', 'r')
lines5 = f5.readlines()
f5.close()
y5=[]
for line in lines5:
    p=line.split()
    y5.append((p[0]))# minimum indexy


#prenik datumov merania k indexu s datumami poruch
match = set(x2).intersection(y1)
matchlist=sorted(list(match))

#zoradenie(nie nutne)
def date_key(a):
            a = datetime.strptime(a, '%d.%m.%Y').date()
            return a
b=sorted_dates = sorted(matchlist, key=date_key)
#print b

if chceme_spolecne==1:
  prvyindex=x2.index(spolecne_t0) #x2 je datum kindexu
  poslednyindex=x2.index(spolecne_t1)
else:
  prvyindex=x2.index(b[0])
  poslednyindex=x2.index(b[len(b)-1])

#zkraceno by MS
f=list(range(prvyindex,poslednyindex))
    
print((len(f)))
print (prvyindex)
print (poslednyindex)

y44=[]
y55=[]

#zo stringu robim float 
int_lstmax = [float(x) for x in y4]
int_lstmin = [float(x) for x in y5]

#len si tvorim integre z y4 maximum
for i in range(len(int_lstmax)):
    y4=int(int_lstmax[i])
    y44.append(y4)

#minimum
for i in range(len(int_lstmin)):
    y5=int(int_lstmin[i])
    y55.append(y5)
    
maxx=(set(f).intersection(y44))
minn=(set(f).intersection(y55))
print((sorted(maxx)))
print((sorted(minn)))

maxx=list(sorted(maxx))
minn=list(sorted(minn))

maximum=[]
minimum=[]

#aby sedeli datumy max a min vzhladom ku datumom poruch
for i in range(len(maxx)):
    novemax=maxx[i]-prvyindex
    print (novemax)
    maximum.append(novemax)
print (maximum)

for i in range(len(minn)):
    novemin=minn[i]-prvyindex
    minimum.append(novemin)
print (minimum)
    
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/sortedmax.txt',(maximum))
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/sortedmin.txt',(minimum))
    

#samotne k indexy na dane datumy
sicke1=[x3[x] for x in range(prvyindex,poslednyindex)]
sicke2=[x4[x] for x in range(prvyindex,poslednyindex)]
sicke3=[x5[x] for x in range(prvyindex,poslednyindex)]
k1=[]
k2=[]
k3=[]
#kindex
for i in range(len(sicke1)):
    k1.append(sicke1[i])

kk1=[]
for i in range (len(k1)):
    d=float(k1[i])
    kk1.append(d)

#prva deriv
for i in range(len(sicke2)):
    k2.append(sicke2[i])

kk2=[]
for i in range (len(k2)):
    d=float(k2[i])
    kk2.append(d)
#print type(kk2[1])
    
#druha deriv
for i in range(len(sicke3)):
    k3.append(sicke3[i])

kk3=[]
for i in range (len(k3)):
    d=float(k3[i])
    kk3.append(d)
#print type(kk3[1])

#datmy=poradie dni
nic=[]
for i in range(len(k1)):
    nic.append(i+1)
#print nic
    

#pocet poruch na dane datumy
for i in b:
    y2.count(i)
    z1.append(y2.count(i))
#print type(z1[1])

print((len(z1)))
print((len(k1)))
print((len(nic)))
    
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/poradie.txt',nic )
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/pocetporuch.txt',z1)
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindexfiltr.txt',kk1)
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/1deriv.txt',kk2)
np.savetxt ('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/2deriv.txt',kk3)

    



            
