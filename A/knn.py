# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
# pylint: disable=maybe-no-member
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def kNearestNeighbors(iris, k):
    model = KNeighborsClassifier(n_neighbors=k)
    # model = model.fit(iris.data, iris.target)
    X = iris.data
    y = iris.target

    model = model.fit(X, y)
    return model

def plotKNearestNeighbors(X, y, xlabel, ylabel, model):
    h = .02  # step size in the mesh

    model.fit(X, y)

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
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (3, 0))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("A/images/knn_" + xlabel + "_" + ylabel)
    # plt.show()
