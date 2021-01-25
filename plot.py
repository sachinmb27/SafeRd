import numpy as np
import scipy.io as si
import progressbar
import re
from filter import butter_lowpass_filter,plotdata
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import norm,multivariate_normal

def normalize(Z,min=-3,max=3):
    scaler = MinMaxScaler(feature_range=(min,max))
    Z = Z.reshape(len(Z),1)
    scaler = scaler.fit(Z)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    normalise = scaler.transform(Z)
    return normalise

def datafeeder(Z,window):
    Zdata = np.array([])
    bar =progressbar.ProgressBar(maxval=len(Z),widgets=[progressbar.Bar('*','[',']'),'',progressbar.Percentage()])
    bar.start()
    for index in range(0,len(Z)-window+1):
        data=np.array(Z[index:index+window])
        Zdata=np.append(Zdata,data)
        Zdata=np.append(Zdata,longitude[index])
        Zdata=np.append(Zdata,latitude[index])
        bar.update(index+1)
    bar.finish()

    Zdata = (Zdata.reshape(index+1,98))
    Zdata = np.hsplit(Zdata,np.array([96]))
    Z = Zdata[0]
    Location = Zdata[1]
    print(Z.shape)
    np.savetxt('filtereddata/predictdata5.txt',Z,fmt='%1.15g')
    print('Data written to File.')

if __name__=='__main__':
    latitude,longitude,X,Y,Z,S,HH,MM,T=np.loadtxt('originaldata/bump10.txt',skiprows=2,unpack=True)
    S = np.around(Z,decimals=2)
    original = Z
    T= HH * 60 * 60 + MM * 60 +T
    X = butter_lowpass_filter(X, cutoff=5,fs= 32,order= 6)
    Y = butter_lowpass_filter(Y, cutoff=5,fs= 32,order= 6)
    Z = butter_lowpass_filter(Z, cutoff=5,fs= 32,order= 6)
    D = np.column_stack((X,Y,Z))
    X_norm = normalize(X,-2,2)
    Y_norm = normalize(Y,-2,2)
    Z_norm = normalize(Z,-2,2)
    delta=norm.logpdf(Z_norm,loc=np.mean(Z_norm,dtype=np.float64,axis=0),scale=
    np.std(Z_norm, dtype=np.float64,axis=0))
    delta3=multivariate_normal.logpdf(D,mean=np.mean(D,
    dtype=np.float64,axis=0),cov=np.std(D, dtype=np.float64,axis=0))
    XYZ_norm = np.column_stack((latitude,longitude,X_norm,Y_norm,Z_norm))
    plotdata(Z_norm,Z,fs=32,t=T)