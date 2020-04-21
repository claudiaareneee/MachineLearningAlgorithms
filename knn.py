# pylint: disable=maybe-no-member
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import graphviz 

iris = datasets.load_iris()

print(iris.data)
print(iris.target)
print(iris.target_names)

model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(iris.data, iris.target)

