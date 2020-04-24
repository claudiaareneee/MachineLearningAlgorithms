from sklearn.cluster import KMeans

def kMeansClustering(iris, numberOfClusters=3):
    kmeans = KMeans(n_clusters=numberOfClusters)
    kmeans.fit(iris.data)
    return kmeans


def plotKMeansClustering(iris, model):
    pass