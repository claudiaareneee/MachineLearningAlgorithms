# pylint: disable=maybe-no-member
from sklearn import datasets
from sklearn.model_selection import train_test_split
import knn as knn
import kmeansclustering as kmeansclustering
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import decisiontree as dtree
import backpropagation as bp
import numpy as np
import copy

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.20)

def makeTestPrediction(model, iris):
    value = [5, 3.4, 1.6, 0.4]
    result = model.predict([ value,])# What is the iris class for 3cm x 5cm sepal and 4cm x 2cm petal?
    print ("%dcm x %dcm sepal and %dcm x %dcm petal: " %tuple(value) + str(iris.target_names[result]))

def plotData(x, y, title):
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(iris.data[:, x], iris.data[:, y], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x])
    plt.ylabel(iris.feature_names[y])
    plt.title(title)
    plt.savefig("A/images/" + title.replace(" ", "-"))
    plt.clf()
    # plt.show()

def plotClf(model, iris, X_train, X_plot, y_train, y_plot, title, plotSepal=True, supervised=True):
    plt.clf()
    h = .02  # step size in the mesh

    if (plotSepal):
        baseIndex = 0
    else: 
        baseIndex = 2  

    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=cmap_bold, edgecolor='k', s=20)
    if not supervised:
        plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='red')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(iris.feature_names[baseIndex])
    plt.ylabel(iris.feature_names[baseIndex + 1])
    plt.savefig("A/images/" + title.replace(" ", "-"))

def plotAll(model, iris, X_train, X_test, y_train, y_test, title, plotSepal=True, supervised=True):
    X_train_Sepal = X_train[:, :2]
    X_test_Sepal = X_test[:, :2]
    X_train_Petal = X_train[:, 2:]
    X_test_Petal = X_test[:, 2:]

    model.fit(X_train_Sepal, y_train)
    plotClf(model, iris, X_train_Sepal, X_train_Sepal, y_train, y_train, title + " Sepal Train Data", plotSepal=True, supervised=supervised)
    plotClf(model, iris, X_train_Sepal, X_test_Sepal, y_train, y_test, title + " Sepal Test Data", plotSepal=True, supervised=supervised)

    model.fit(X_train_Petal, y_train)
    plotClf(model, iris, X_train_Petal, X_train_Petal, y_train, y_train, title + " Petal Train Data", plotSepal=False, supervised=supervised)
    plotClf(model, iris, X_train_Petal, X_test_Petal, y_train, y_test, title + " Petal Test Data", plotSepal=False, supervised=supervised)    

if __name__ == "__main__":
    print("Iris data:")
    print(iris.data)
    print("Iris target:")
    print(iris.target)
    print("Iris target names:")
    print(iris.target_names)

    print("X_train:")
    print(X_train)
    print("X_test:")
    print(X_test)
    print("y_train:")
    print(y_train)
    print("y_test:")
    print(y_test)

    plotData(0,1,"Iris classification according to Sepal measurements")
    plotData(2,3,"Iris classification according to Petal measurements")

    print("Performing Decision Tree             ", end="    ")
    model = dtree.decisionTree(X_train, y_train)
    makeTestPrediction(model, iris)
    dtree.plotDecisionTree(model)
    plotAll(model, iris, X_train, X_test, y_train, y_test, "Decision Tree", plotSepal=False)

    print("Performing Decision Tree Max Depth 4 ", end="    ")
    model = dtree.decisionTree(X_train, y_train, max_depth=4)
    makeTestPrediction(model, iris)
    plotAll(model, iris, X_train, X_test, y_train, y_test, "Decision Tree Max Depth 4", plotSepal=False)

    print("Performing K Nearest Neighbors       ", end="    ")
    model = knn.kNearestNeighbors(X_train, y_train, 3)
    makeTestPrediction(model, iris)
    plotAll(model, iris, X_train, X_test, y_train, y_test, "K Nearest Neighbors", plotSepal=False)

    print("Performing K Means Clustering        ", end="    ")
    model = kmeansclustering.kMeansClustering(X_train, numberOfClusters=3)
    makeTestPrediction(model, iris) # This is off because the index of the clusters don't match
    plotAll(model, iris, X_train, X_test, y_train, y_test, "K Means Clustering", plotSepal=False, supervised=False)
    
    print("Performing Back propagation          ", end="    ")
    model = bp.backPropagation(X_train, X_test, y_train, y_test)
    makeTestPrediction(model, iris)
    plotAll(model, iris, X_train, X_test, y_train, y_test, "Back Propagation", plotSepal=False)

# TODO: k Means Clustering