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

irisSplit = copy.deepcopy(iris)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.20)
irisSplit.data = X_train
irisSplit.target = y_train

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

def plotClf(model, iris, title, plotSepal=True):
    plt.clf()
    h = .02  # step size in the mesh

    if (plotSepal):
        baseIndex = 0
        X = iris.data[:, :2]
        y = iris.target
    else: 
        baseIndex = 2
        X = iris.data[:, 2:]
        y = iris.target

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
    # plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
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

    print("IrisSplit data:")
    print(irisSplit.data)
    print("IrisSplit target:")
    print(irisSplit.target)
    print("IrisSplit target names:")
    print(irisSplit.target_names)

    plotData(0,1,"Iris classification according to Sepal measurements")
    plotData(2,3,"Iris classification according to Petal measurements")


    print("Performing Decision Tree             ", end="    ")
    model = dtree.decisionTree(iris)
    makeTestPrediction(model, iris)
    dtree.plotDecisionTree(iris, model)
    plotClf(model, iris, "Decision Tree Petal", plotSepal=False)
    plotClf(model, iris, "Decision Tree Sepal", plotSepal=True)

    print("Performing Decision Tree Max Depth 4 ", end="    ")
    model = dtree.decisionTree(iris, max_depth=4)
    makeTestPrediction(model, iris)
    plotClf(model, iris, "Decision Tree Petal Max depth 4", plotSepal=False)
    plotClf(model, iris, "Decision Tree Sepal Max depth 4", plotSepal=True)

    print("Performing K Nearest Neighbors       ", end="    ")
    model = knn.kNearestNeighbors(iris, 3)
    makeTestPrediction(model, iris)
    plotClf(model, iris, "Knn Petal", plotSepal=False)
    plotClf(model, iris, "Knn Sepal", plotSepal=True)

    print("Performing K Means Clustering        ", end="    ")
    model = kmeansclustering.kMeansClustering(iris, numberOfClusters=3)
    makeTestPrediction(model, iris) # This is off because the index of the clusters don't match
    kmeansclustering.plotKMeansClustering(model, iris, "K Means Clustering Petal", plotSepal=False)
    kmeansclustering.plotKMeansClustering(model, iris, "K Means Clustering Sepal", plotSepal=True)
    
    print("Performing Back propagation          ", end="    ")
    model = bp.backPropagation(X_train, X_test, y_train, y_test)
    makeTestPrediction(model, iris)
    plotClf(model, iris, "Back propagation Sepal", plotSepal=True)
    plotClf(model, iris, "Back propagation Petal", plotSepal=False)

    print("Split Performing Decision Tree             ", end="    ")
    model = dtree.decisionTree(irisSplit)
    makeTestPrediction(model, irisSplit)
    dtree.plotDecisionTree(irisSplit, model)
    plotClf(model, irisSplit, "Split Decision Tree Petal", plotSepal=False)
    plotClf(model, irisSplit, "Split Decision Tree Sepal", plotSepal=True)

    print("Split Performing Decision Tree Max Depth 4 ", end="    ")
    model = dtree.decisionTree(irisSplit, max_depth=4)
    makeTestPrediction(model, irisSplit)
    plotClf(model, irisSplit, "Split Decision Tree Petal Max depth 4", plotSepal=False)
    plotClf(model, irisSplit, "Split Decision Tree Sepal Max depth 4", plotSepal=True)

    print("Split Performing K Nearest Neighbors       ", end="    ")
    model = knn.kNearestNeighbors(irisSplit, 3)
    makeTestPrediction(model, irisSplit)
    plotClf(model, irisSplit, "Split Knn Petal", plotSepal=False)
    plotClf(model, irisSplit, "Split Knn Sepal", plotSepal=True)

    print("Split Performing K Means Clustering        ", end="    ")
    model = kmeansclustering.kMeansClustering(irisSplit, numberOfClusters=3)
    makeTestPrediction(model, irisSplit) # This is off because the index of the clusters don't match
    kmeansclustering.plotKMeansClustering(model, irisSplit, "Split K Means Clustering Petal", plotSepal=False)
    kmeansclustering.plotKMeansClustering(model, irisSplit, "Split K Means Clustering Sepal", plotSepal=True)
    
    print("Split Performing Back propagation          ", end="    ")
    model = bp.backPropagation(X_train, X_test, y_train, y_test)
    makeTestPrediction(model, irisSplit)
    plotClf(model, irisSplit, "Split Back propagation Sepal", plotSepal=True)
    plotClf(model, irisSplit, "Split Back propagation Petal", plotSepal=False)

# TODO: k Means Clustering