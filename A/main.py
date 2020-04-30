# pylint: disable=maybe-no-member
# Sources:  http://stephanie-w.github.io/brainscribble/classification-algorithms-on-iris-dataset.html, 
#           https://constantgeeks.com/2017/01/11/playing-with-iris-data-kmeans-clustering-in-python/, 
#           https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/,
#           https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342
from sklearn import datasets, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import knn as knn
import kmeansclustering as kmeansclustering
import decisiontree as dtree
import backpropagation as bp
import numpy as np

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size = 0.20)

global file

def makeTestPrediction(model, iris):
    value = [5, 3.4, 1.6, 0.4]
    result = model.predict([ value,])# What is the iris class for 3cm x 5cm sepal and 4cm x 2cm petal?
    file.write ("%dcm x %dcm sepal and %dcm x %dcm petal: " %tuple(value) + str(iris.target_names[result]))
    file.write("\n")

def plotData(x, y, title):
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
    plt.scatter(iris.data[:, x], iris.data[:, y], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x])
    plt.ylabel(iris.feature_names[y])
    plt.title(title)
    plt.savefig("A/images/" + title.replace(" ", "-"))
    plt.clf()

def plotClf(model, iris, X_train, X_plot, y_train, y_plot, title, plotSepal=True, supervised=True):
    plt.clf()
    h = .02  # step size in the mesh

    if (plotSepal):
        baseIndex = 0
    else: 
        baseIndex = 2  

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
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=cmap_bold, edgecolor='k', s=20)
    if not supervised:
        plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='red')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(iris.feature_names[baseIndex])
    plt.ylabel(iris.feature_names[baseIndex + 1])
    plt.savefig("A/images/" + title.replace(" ", ""))

def plotAll(model, iris, X_train, X_test, y_train, y_test, title, plotSepal=True, supervised=True):
    X_train_Sepal = X_train[:, :2]
    X_test_Sepal = X_test[:, :2]
    X_train_Petal = X_train[:, 2:]
    X_test_Petal = X_test[:, 2:]

    model.fit(X_train_Sepal, y_train)
    plotClf(model, iris, X_train_Sepal, X_train_Sepal, y_train, y_train, title + " Sepal Training Data", plotSepal=True, supervised=supervised)
    plotClf(model, iris, X_train_Sepal, X_test_Sepal, y_train, y_test, title + " Sepal Test Data", plotSepal=True, supervised=supervised)

    model.fit(X_train_Petal, y_train)
    plotClf(model, iris, X_train_Petal, X_train_Petal, y_train, y_train, title + " Petal Training Data", plotSepal=False, supervised=supervised)
    plotClf(model, iris, X_train_Petal, X_test_Petal, y_train, y_test, title + " Petal Test Data", plotSepal=False, supervised=supervised)    

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
    # file.write("Iris data:")
    # file.write(iris.data)
    # file.write("Iris target:")
    # file.write(iris.target)
    # file.write("Iris target names:")
    # file.write(iris.target_names)

    # file.write("X_train:")
    # file.write(X_train)
    # file.write("X_test:")
    # file.write(X_test)
    # file.write("y_train:")
    # file.write(y_train)
    # file.write("y_test:")
    # file.write(y_test)

    file = open('metricsA.txt', 'w')

    plotData(0,1,"Iris classification according to Sepal measurements")
    plotData(2,3,"Iris classification according to Petal measurements")

    models = []

    models.append(("Decision Tree", dtree.decisionTree(X_train, y_train)))
    models.append(("Decision Tree Max Depth 4", dtree.decisionTree(X_train, y_train, max_depth=4)))
    models.append(("K Nearest Neighbors", knn.kNearestNeighbors(X_train, y_train, 3)))
    models.append(("K Means Clustering", kmeansclustering.kMeansClustering(X_train, numberOfClusters=3)))
    models.append(("Back propagation", bp.backPropagation(X_train, X_test, y_train, y_test)))

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

        makeTestPrediction(model, iris)
        spotCheck(name, model)
        plotAll(model, iris, X_train, X_test, y_train, y_test, name, plotSepal=False, supervised=supervised)
    
    file.close()

# TODO: k Means Clustering