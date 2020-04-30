# pylint: disable=maybe-no-member
from sklearn import datasets, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import knn as knn
import kmeansclustering as kmeansclustering
import decisiontree as dtree
import backpropagation as bp
import numpy as np

# Load in the `digits` data
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = model_selection.train_test_split(digits.data, digits.target, test_size = 0.20)

global file

def makeTestPrediction(model, iris):
    value = [5, 3.4, 1.6, 0.4]
    result = model.predict([ value,])# What is the iris class for 3cm x 5cm sepal and 4cm x 2cm petal?
    file.write ("%dcm x %dcm sepal and %dcm x %dcm petal: " %tuple(value) + str(iris.target_names[result]))
    file.write("\n")

def spotCheck(name, model):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    file.write("Test accuracy score: %f\n" % score)
    file.write("Confusion matrix:\n")
    score = confusion_matrix(y_test, predictions)
    file.write(str(score))
    file.write("\n")
    file.write("Classification report:\n")
    score  = classification_report(y_test, predictions)
    file.write(score)
    file.write("\n")

    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    msg = "Training accuracy score: %f (%f)\n" % (cv_results.mean(), cv_results.std())
    file.write(msg)

if __name__ == "__main__":
    file = open('metricsAExtraCredit.txt', 'w')

    models = []

    models.append(("Extra Credit Decision Tree", dtree.decisionTree(X_train, y_train)))
    models.append(("Extra Credit Decision Tree Max Depth 4", dtree.decisionTree(X_train, y_train, max_depth=4)))
    models.append(("Extra Credit K Nearest Neighbors", knn.kNearestNeighbors(X_train, y_train, 3)))
    models.append(("Extra Credit K Means Clustering", kmeansclustering.kMeansClustering(X_train, numberOfClusters=3)))
    models.append(("Extra Credit Back propagation", bp.backPropagation(X_train, X_test, y_train, y_test)))

    # dtree.plotDecisionTree(model)

    for name, model in models:
        file.write("---------------------------------------------------------------------------------------------\n")
        file.write(name)
        file.write("\n")

        supervised = True
        if (name is "Decision Tree"):
            dtree.plotDecisionTree(model, name)
        elif ("K Means Clustering" in name):
            supervised = False

        spotCheck(name, model)
    
    file.close()

# TODO: k Means Clustering