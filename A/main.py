# pylint: disable=maybe-no-member
from sklearn import datasets
import knn as knn
import decisiontree as dtree
import backpropagation as bp

iris = datasets.load_iris()

print(iris.data)
print(iris.target)
print(iris.target_names)

print("Performing Decision Tree")
model = dtree.decisionTree(iris)
dtree.plotDecisionTree(iris, model)

print("Performing K Nearest Neighbors")
model = knn.kNearestNeighbors(iris, 3)
knn.plotKNearestNeighbors(iris, model)

print("Performing Back propagation")
model = bp.backPropagation(iris)
bp.plotBackPropagation(iris, model)

# TODO: k Means Clustering