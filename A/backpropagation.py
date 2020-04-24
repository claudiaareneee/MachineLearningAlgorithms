from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def backPropagation(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.ravel()) # This was y_train.values.ravel()
    return mlp

def plotBackPropagation(iris, model):
    # y_test, predictions = model
    # print("Confusion Matrix: ")
    # print(confusion_matrix(y_test,predictions))

    # print("")

    # print("Classification Report: ")
    # print(classification_report(y_test,predictions))

    pass