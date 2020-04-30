# Sources: https://constantgeeks.com/2017/01/11/playing-with-iris-data-kmeans-clustering-in-python/
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.cluster import KMeans

def kMeansClustering(X_train, numberOfClusters=3):
    kmeans = KMeans(n_clusters=numberOfClusters)
    kmeans.fit(X_train)
    return kmeans