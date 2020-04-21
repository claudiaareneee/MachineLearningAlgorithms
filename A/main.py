from sklearn import datasets
import knn as knn

iris = datasets.load_iris()

model = knn.kNearestNeighbors(iris, 3)
knn.plotKNearestNeighbors(iris, model)