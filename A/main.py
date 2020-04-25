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

def plotClf(model, iris, X_train, y_train, title, plotSepal=True, supervised=True):
    plt.clf()
    h = .02  # step size in the mesh

    if (plotSepal):
        baseIndex = 0
        X = X_train[:, :2]
    else: 
        baseIndex = 2
        X = X_train[:, 2:]

    if supervised:
        model.fit(X, y_train)
    else: 
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
    plt.scatter(X[:, 0], X[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    if not supervised:
        plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='red')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(iris.feature_names[baseIndex])
    plt.ylabel(iris.feature_names[baseIndex + 1])
    plt.savefig("A/images/" + title.replace(" ", "-"))
    # plt.show()

def plotClfTestData(model, iris, X_train, X_test, y_train, y_test, title, plotSepal=True, supervised=True):
    plt.clf()
    h = .02  # step size in the mesh

    if (plotSepal):
        baseIndex = 0
        X_train = X_train[:, :2]
        X_test = X_test[:, :2]
    else: 
        baseIndex = 2
        X_train = X_train[:, 2:]
        X_test = X_test[:, 2:]

    model.fit(X_train, y_train)    

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
    # plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    if not supervised:
        plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='red')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(iris.feature_names[baseIndex])
    plt.ylabel(iris.feature_names[baseIndex + 1])
    plt.savefig("A/images/" + title.replace(" ", "-"))
    # plt.show()

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
    plotClf(model, iris, X_train, y_train, "Decision Tree Petal", plotSepal=False)
    plotClf(model, iris, X_train, y_train, "Decision Tree Sepal", plotSepal=True)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Decision Tree Petal Test Data", plotSepal=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Decision Tree Sepal Test Data", plotSepal=True)

    print("Performing Decision Tree Max Depth 4 ", end="    ")
    model = dtree.decisionTree(X_train, y_train, max_depth=4)
    makeTestPrediction(model, iris)
    plotClf(model, iris, X_train, y_train, "Decision Tree Petal Max depth 4", plotSepal=False)
    plotClf(model, iris, X_train, y_train, "Decision Tree Sepal Max depth 4", plotSepal=True)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Decision Tree Petal Max depth 4 Test Data", plotSepal=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Decision Tree Sepal Max depth 4 Test Data", plotSepal=True)

    print("Performing K Nearest Neighbors       ", end="    ")
    model = knn.kNearestNeighbors(X_train, y_train, 3)
    makeTestPrediction(model, iris)
    plotClf(model, iris, X_train, y_train, "Knn Petal", plotSepal=False)
    plotClf(model, iris, X_train, y_train, "Knn Sepal", plotSepal=True)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Knn Petal Test Data", plotSepal=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Knn Sepal Test Data", plotSepal=True)

    print("Performing K Means Clustering        ", end="    ")
    model = kmeansclustering.kMeansClustering(X_train, numberOfClusters=3)
    makeTestPrediction(model, iris) # This is off because the index of the clusters don't match
    plotClf(model, iris, X_train, y_train, "K Means Clustering Petal", plotSepal=False, supervised=False)
    plotClf(model, iris, X_train, y_train, "K Means Clustering Sepal", plotSepal=True, supervised=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "K Means Clustering Petal Test Data", plotSepal=False, supervised=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "K Means Clustering Sepal Test Data", plotSepal=True, supervised=False)
    
    print("Performing Back propagation          ", end="    ")
    model = bp.backPropagation(X_train, X_test, y_train, y_test)
    makeTestPrediction(model, iris)
    plotClf(model, iris, X_train, y_train, "Back propagation Sepal", plotSepal=True)
    plotClf(model, iris, X_train, y_train, "Back propagation Petal", plotSepal=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Back propagation Petal Test Data", plotSepal=False)
    plotClfTestData(model, iris, X_train, X_test, y_train, y_test, "Back propagation Sepal Test Data", plotSepal=True)

# TODO: k Means Clustering