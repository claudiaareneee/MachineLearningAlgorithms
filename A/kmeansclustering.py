import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.cluster import KMeans

def kMeansClustering(iris, numberOfClusters=3):
    kmeans = KMeans(n_clusters=numberOfClusters)
    kmeans.fit(iris.data)
    return kmeans

def plotKMeansClustering(model, iris, title, plotSepal=True):
    plt.clf()
    h = .02  # step size in the mesh

    if (plotSepal):
        baseIndex = 0
        X = iris.data[:, :2]
    else: 
        baseIndex = 2
        X = iris.data[:, 2:]

    y = iris.target

    model.fit(X)

    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='red')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(iris.feature_names[baseIndex])
    plt.ylabel(iris.feature_names[baseIndex + 1])
    plt.savefig("A/images/" + title.replace(" ", "-"))
    # plt.show()