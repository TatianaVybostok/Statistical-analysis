from datetime import date, datetime, timedelta
import julian
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]

#budkov
#fhandle = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/BDV_K-index.txt', 'r')
#hurbanovo
fhandle = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k_hurb.txt', 'r')
lines = fhandle.readlines()
fhandle.close()

date = []
kindex=[]
for line in lines:
    p = line.split()
    date.append(str(p[1]))
    #budkov
    #kindex.append(float(p[9]))
    #hurb
    kindex.append(float(p[2]))

kindex_arr=np.array(kindex)

date2=[]
for i in date:
    #budkov
    #date_.append(datetime.strptime(i, '%d.%m.%Y'))
    #hurbanovo
    date2.append(datetime.strptime(i, '%Y-%m-%d'))
#hurbanovo
date3=[]
for i in date2:
    date3.append(i.strftime('%d.%m.%Y'))
print(date3[0:10])
#hurbanovo
date_=[]
for d in date3:
    date_.append(datetime.strptime(d,'%d.%m.%Y'))

print(type(date_),date_[0:10])

jd=[]
for j in date_:
    jd.append(julian.to_jd(j, fmt='jd'))

N=len(jd)
dT=24.*60.*60.
X = np.arange((N - 1)/2) + 1
is_N_even = (N % 2) == 0
freq=np.zeros(N)
if is_N_even:
    for i in range(1,N//2):
        freq[i]=i/(dT*N)
        freq[N-i]=-i/(dT*N)
else:
    for i in range(1,(N-1)//2):
        freq[i]=i/(dT*N)
        freq[N-i]=-i/(dT*N)

#overenie s IDL
#x =  np.array([-2., 8., -6., 4., 1., 0., 3., 5.])
#y=np.fft.fft(x)
#print(y/len(y)) #( 1.62500, 0.00000) ( 0.420495, 0.506282) ( 0.250000, 0.125000) ( -1.17050, -1.74372) ( -2.62500, -0.00000) ( -1.17050, 1.74372) ( 0.250000, -0.125000) ( 0.420495, -0.506282)

def boxcarsmooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

for window in [10,20, 30, 40, 50, 60, 70, 80, 90, 100]:

    #kindex_w=boxcarsmooth(kindex_arr,window)
    kindex_w=gaussian_filter1d( kindex_arr,window)
    #kindex_w=kindex_arr
    
    plt.plot(kindex_w, 'b', label='filtered, sigma='+str(window))
    plt.legend()
    plt.grid()
    #plt.show()
    plt.close()

    der=(np.diff(kindex_w))

    #FFT
    day2sec=24.*60.*60.0

    const=1.0j * (2*np.pi)* freq 

    ft_kindex=np.fft.fft(kindex_w)

    fd = np.real(np.fft.ifft(const * (ft_kindex)))
    fd=boxcarsmooth(fd,4)*day2sec*12
    fd2 = np.real(np.fft.ifft( 1.0j * freq * np.fft.fft(fd)))
    fd2=boxcarsmooth(fd2,4)*day2sec*12*12

    poradie=[]
    for i in range (len(kindex_w)):poradie.append(i+1)

    #print(np.corrcoef(der,fd[1:]))
    #plt.plot(np.correlate(der,fd))
    #plt.plot(kindex)
    fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    ax1.plot(der)
    ax1.grid()
    ax1.legend()
    ax2.plot(fd[10:5550])
    ax2.grid()
    ax2.legend()
    #plt.show()
    plt.close()

    np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k-' + str(window)+'.txt', np.array([poradie,date3,kindex_w, fd, fd2]).T, delimiter='\t', fmt="%s")
    #np.savetxt('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k_der.txt', np.array([poradie,date3,kindex_w, fd, fd2]).T, delimiter='\t', fmt="%s")
    
    #print(1)   
     
        
fhandle1 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/filtered/filtered_k-10.txt', 'r')
lines1 = fhandle1.readlines()
fhandle1.close()

poradie1=[]
date1 = []
kindex1=[]
for line in lines1:
    p = line.split()
    poradie1.append(int(p[0]))
    date1.append(str(p[1]))
    kindex1.append(float(p[2]))

print(date[0],date1[0],date[-1],date1[-1])

first_ind=[]

for i,item in enumerate(date1):
    if item == '01.01.2009':
        first_ind=i
    else:
        pass
#last_ind=5477 #320 nT
last_ind=5478 #420nT

date1=date1[first_ind:last_ind]
print(date[0],date1[0],date[-1],date1[-1])


fhandle2 = open('/Users/Taninka/Documents/skola_PhD/STATISTICAL_ANALYSIS/SEPS/data/kindex_hurb_420nT/k_value/k-10.txt', 'r')
lines2 = fhandle2.readlines()
fhandle2.close()

poradie2=[]
date2 = []
kindex2=[]
for line in lines2:
    p = line.split()
    poradie2.append(int(p[0]))
    date2.append(str(p[1]))
    kindex2.append(float(p[2]))

kindex1=kindex1[first_ind:last_ind]


fig, ((ax1), (ax2),(ax3)) = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
ax1.plot(kindex1,label='Michal')
ax1.grid()
ax1.legend()
ax2.plot(kindex2,'r',label='Tatiana')
ax2.grid()
ax2.legend()
ax3.plot(gaussian_filter1d (kindex_w,10),'k',label='Fourierova derivace',linewidth=0.5)
ax3.grid()
ax3.legend()
plt.show()
plt.close()
