from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt

X, Y, Z, Latitude, Longitude = np.loadtxt('filtereddata/XYZ_norm2.txt', skiprows=0, unpack=True)
Location = np.column_stack((Latitude, Longitude))
Data = np.column_stack((X, Y, Z))
dbscan = DBSCAN(eps = 0.874, algorithm='kd_tree', min_samples=2).fit(Data)
print(dbscan.labels_)
print(Data.shape)
pred = np.column_stack((Data, dbscan.labels_))
print(pred.shape)
print('Prediction data ready')