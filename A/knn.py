# Sources: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html and https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# pylint: disable=maybe-no-member
from sklearn.neighbors import KNeighborsClassifier

def kNearestNeighbors(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)

    model = model.fit(X_train, y_train)
    return model
