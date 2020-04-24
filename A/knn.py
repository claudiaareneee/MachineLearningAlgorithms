# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
# pylint: disable=maybe-no-member
from sklearn.neighbors import KNeighborsClassifier

def kNearestNeighbors(iris, k):
    model = KNeighborsClassifier(n_neighbors=k)
    # model = model.fit(iris.data, iris.target)
    X = iris.data
    y = iris.target

    model = model.fit(X, y)
    return model
