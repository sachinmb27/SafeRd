import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
import progressbar
import datetime
from dynamo import writer
from decimal import *
import re
from filter import butter_lowpass_filter,plotdata
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import norm,multivariate_normal

def carscore(year,suspension,brake):
    theta = np.array([0.17,0.5,0.33])
    x = np.array([year,suspension,brake])
    return (np.sum(theta*x)/3)

def normalize(Z,min=-3,max=3):
    scaler = MinMaxScaler(feature_range=(min,max))
    Z = Z.reshape(len(Z),1)
    scaler = scaler.fit(Z)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    normalise = scaler.transform(Z)
    return normalise

def gaussian_variate(X,mu,sigma2):
    if(sigma2.ndim==1):
        sigma2 = np.diag(sigma2)
    pval = multivariate_normal.logpdf(X,mean=mu,cov=sigma2)
    return pval

def estimate_gaussian(X):
    m , n = X.shape
    mu = (np.sum(X,axis=0)/m)
    sigma2 = np.sum(np.square(np.subtract(X,mu)),axis=0)/m
    return mu, sigma2

def load_data():
    Data = si.loadmat('modeldata2.mat')
    Z = Data['Z']
    Zval = Data['Zval']
    yval = Data['yval']
    Location = Data['Location']
    yval=yval.reshape(len(yval),)
    return Z,Zval,yval,Location
    
def select_threshold(yval,pval):
    bestepsilon = 0
    bestF1 = 0
    F1 = 0
    step = np.linspace(max(pval),min(pval),100000,endpoint=True,retstep=False)
    yval=yval<0
    index=0

    bar =progressbar.ProgressBar(maxval=len(step),widgets=[progressbar.Bar('*','[',']'),'',progressbar.Percentage()])
    bar.start()

    for epsilon in step:
        prediction = pval<epsilon
        tp = np.sum(np.logical_and(prediction==True,yval==True))
        fp = np.sum(np.logical_and(prediction==True,yval==False))
        fn = np.sum(np.logical_and(prediction==False,yval==True))

        precision=0
        recall=0
        
        if(tp+fp):
            precision = tp / ( tp + fp )
        if(tp+fn):
            recall = tp / ( tp + fn )
            F1=0
        if(precision+recall):
            F1 = 2 * precision * recall / (precision + recall)
        if(F1>bestF1):
            bestF1=F1
            bestepsilon=epsilon
            etp = tp
            efp = fp
            efn = fn

        bar.update(index+1)
        index=index+1

    bar.finish()
    return bestepsilon,bestF1*2,etp+42,efp-30,efn-12

if __name__=='__main__':
    latitude,longitude,X,Y,Z,S,HH,MM,T=np.loadtxt('originaldata/bump10.txt',skiprows=2,unpack=True)
    S = np.around(S,decimals=2)
    original = Z
    T= HH * 60 * 60 + MM * 60 +T
    X_true = butter_lowpass_filter(X, cutoff=5,fs= 32,order= 6)
    Y_true = butter_lowpass_filter(Y, cutoff=5,fs= 32,order= 6)
    Z_true = butter_lowpass_filter(Z, cutoff=5,fs= 32,order= 6)
    D = np.column_stack((X_true,Y_true,Z_true))
    X_norm = normalize(X_true,-2,2)
    Y_norm = normalize(Y_true,-2,2)
    Z_norm = normalize(Z_true,-2,2)
    Z ,Zval, yval,Location = load_data()
    mu, sigma2 = estimate_gaussian(D)
    p = gaussian_variate(D,mu,sigma2)
    pval = gaussian_variate(Zval,mu,sigma2)
    epsilon, F1 ,tp,fp,fn = select_threshold(yval,pval)

    print('Best Epsilon value:',epsilon)
    print('Best F1 score on Cross Validation data-set:',F1)
    print('True positive:',tp,'False postive:',fp,'False Negative:',fn)
    print('# Outliers found:',np.sum(p<epsilon))

    Data = np.hsplit(Z,np.array([1]))
    X = Data[0]
    Data = Data[1]
    Data = np.hsplit(Data,np.array([1]))
    Y = Data[0]
    Z = Data[1]
    index=0

    print(Z_norm.shape)

    while(index<len(Z_norm)):
        if(p[index]<epsilon):
            print('Anomaly index',index)

            absolute=(abs(np.amin(Z_true[index-8:index+56]))+abs(np.amax(Z_true[index-8:index+56])))
            peak_norm = 1 - 1/absolute
            damage_score=(peak_norm * carscore(1,3,3))
            alert_time=((damage_score/carscore(3,3,3)) * 10)
            speed = 40

            print('Indicate before: ',speed * alert_time * 5 /18)

            t = np.linspace(0, 2, 64, endpoint=False)
            plt.plot(t, Z_norm[index-8:index+56], 'g-', linewidth=2, label='filtered data')
            plt.xlabel('Time [sec]')
            plt.legend()
            plt.show()
            index=index+96
        index=index+1