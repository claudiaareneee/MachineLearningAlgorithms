# pylint: disable=maybe-no-member
from sklearn import datasets
import knn as knn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import decisiontree as dtree
import backpropagation as bp

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

if __name__ == "__main__":
    print(iris.data)
    print(iris.target)
    print(iris.target_names)

    plotData(0,1,"Iris classification according to Sepal measurements")
    plotData(2,3,"Iris classification according to Petal measurements")


    print("Performing Decision Tree")
    model = dtree.decisionTree(iris)
    dtree.plotDecisionTree(iris, model)

    print("Performing K Nearest Neighbors")
    model = knn.kNearestNeighbors(iris, 3)
    knn.plotKNearestNeighbors(iris.data[:, :2], iris.target, iris.feature_names[0], iris.feature_names[1], model)
    knn.plotKNearestNeighbors(iris.data[:, 2:], iris.target, iris.feature_names[2], iris.feature_names[3], model)

    print("Performing Back propagation")
    model = bp.backPropagation(iris)
    bp.plotBackPropagation(iris, model)

# TODO: k Means Clustering