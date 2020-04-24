# pylint: disable=maybe-no-member
from sklearn import datasets
import knn as knn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import decisiontree as dtree
import backpropagation as bp
import numpy as np

iris = datasets.load_iris()

def plotData(x, y, title):
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(iris.data[:, x], iris.data[:, y], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x])
    plt.ylabel(iris.feature_names[y])
    plt.title(title)
    plt.savefig("A/images/" + title)
    plt.clf()
    # plt.show()

def plotClf(model, iris, title, plotSepal):
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
    plt.savefig("A/images/" + title)
    # plt.show()

if __name__ == "__main__":
    print(iris.data)
    print(iris.target)
    print(iris.target_names)

    plotData(0,1,"Iris classification according to Sepal measurements")
    plotData(2,3,"Iris classification according to Petal measurements")


    print("Performing Decision Tree")
    model = dtree.decisionTree(iris)
    # dtree.plotDecisionTree(iris, model)
    plotClf(model, iris, "Decision Tree Petal", False)
    plotClf(model, iris, "Decision Tree Sepal", True)

    print("Performing K Nearest Neighbors")
    model = knn.kNearestNeighbors(iris, 3)
    plotClf(model, iris, "Knn Petal", False)
    plotClf(model, iris, "Knn Sepal", True)

    # print("Performing Back propagation")
    # model = bp.backPropagation(iris)
    # bp.plotBackPropagation(iris, model)

# TODO: k Means Clustering