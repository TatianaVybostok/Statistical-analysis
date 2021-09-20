# definice funkce pro cteni souboru
from datetime import datetime

def parsefile(filename):
  fhandle=open(filename,'r')
  lines=fhandle.readlines()
  fhandle.close()
  retarr=[]
  for line in lines:
    p=line.split()
    retarr.append((p[0]))
  return retarr

# pres vsechny distributory hledam data prekryvu
ndistributors=1
t0=[]
t1=[]
for distributor in range(ndistributors):
  file_datumy='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-'+str(distributor+1)+'-datumy.txt'
  l=parsefile(file_datumy)
  t0.append(l[0])
  t1.append(l[-1])

def date_key(a):
            a = datetime.strptime(a, '%d.%m.%Y').date()
            return a
t0 = sorted(t0, key=date_key)
t1 = sorted(t1, key=date_key)

spolecne_t0=t0[-1]
spolecne_t1=t1[0]

chceme_spolecne=0

flog= open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/tables/table_SEPS_hurb_modeled_test.tex',"w+") #jenda tbaulka pre vsetky okna

#flog= open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/tables/table_SEPS_modeled_test.tex',"w+") #jenda tbaulka pre vsetky okna
flog.write('Window & Intervals & $n_H$ & $n_L$ & $p_{HL}$ & $R$ \\\\ \n')
flog.write('\\hline \n')

for window in [10,20,30,40,50,60,70,80,90,100]:#,120,150,200]:
#window=70
  #budkov
  #filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/filtered/filtered_k-' + str(window)+'.txt'

  #hurbanovo
  filein='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k-' + str(window)+'.txt'

  exec(open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/code/intervaly2.py').read())

  #flog= open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/tables/table-'+str(window)+'.tex',"w+") tabulky pre ejdnotlive okna 
  #flog.write('Window & Dataset & Intervals & $n_H$ & $n_R$ & $n_L$ & $p_{HL}$ & $p_{RL}$ & $p_{RH}$ & $R$ \\\\ \n')
  #flog.write('\\hline \n')

  #distributor=1

  for distributor in range(1,ndistributors+1):
    file_datumy='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-'+str(distributor)+'-datumy.txt'
    file_datumy_poruch='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-'+str(distributor)+'-poruchy_rezid_test.txt'

    file_ID='/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/distributori/dataset-'+str(distributor)+'-ID.txt'
    f1 = open(file_ID, 'r')
    lines1 = f1.readlines()
    f1.close()
    ID=lines1[0].rstrip()

    exec(open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/code/datumy2.py').read())

    exec(open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/code/MAIN2.py').read())

  #flog.close()
flog.close()

