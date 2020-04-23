from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def backPropagation(iris):
    # # Assign data from first four columns to X variable
    # X = iris.data
    # # Assign data from first fifth columns to y variable
    # y = iris.target
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # scaler = StandardScaler()
    # scaler.fit(X_train)

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    # mlp.fit(X_train, y_train.values.ravel())

    # predictions = mlp.predict(X_test) # returns predictions instead of model rn.

    # return (y_test, predictions)
    pass


def plotBackPropagation(iris, model):
    # y_test, predictions = model
    # print("Confusion Matrix: ")
    # print(confusion_matrix(y_test,predictions))

    # print("")

    # print("Classification Report: ")
    # print(classification_report(y_test,predictions))

    pass