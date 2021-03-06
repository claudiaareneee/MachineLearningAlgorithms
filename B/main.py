# Sources: https://wiseodd.github.io/techblog/2016/06/21/nn-sgd/ and https://github.com/dennybritz/nn-from-scratch/blob/0b52553c84c8bd5fed4f0c890c98af802e9705e9/nn_from_scratch.py
from sklearn.datasets import make_moons, load_iris, make_classification
from sklearn import datasets, model_selection
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
import time

n_feature = 2
n_class = 2
n_iter = 100

global file

def plotData(x, y, title):
    plt.scatter(x[:,0], x[:,1], c=y, edgecolor='k', s=20)
    # plt.xlabel(iris.feature_names[x])
    # plt.ylabel(iris.feature_names[y])
    # plt.title(title)
    plt.savefig("B/images/" + title.replace(" ", "-"))
    plt.clf()

def make_network(n_hidden=100, n_feature=2, n_class=2):
    # Initialize weights with Standard Normal random variables
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )

    return model

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    # ReLU non-linearity
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h @ model['W2'])

    return h, prob

def backward(model, xs, hs, errs):
    """xs, hs, errs contain all informations (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    dW2 = hs.T @ errs

    # Get gradient of hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0

    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)

def sgd(model, X_train, y_train, minibatch_size):
    times = []

    overall_time = time.time()
    
    for iter in range(n_iter):
        # file.write('Iteration {}\n'.format(iter))

        initial_time = time.time()

        # Randomize data point
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

            model = sgd_step(model, X_train_mini, y_train_mini)
        
        times.append(time.time() - initial_time)

    times = np.array(times)
    file.write("Mean iteration time: " + str(times.mean()) + "\n")
    file.write("Total time taken: " + str(time.time() - overall_time) + "\n")
    return model

def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Learning rate: 1e-4
        model[layer] += 1e-4 * grad[layer]

    return model

def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        # Create probability distribution of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.

        # Compute the gradient of output layer
        err = y_true - y_pred

        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))

def doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size = 50, title="Gradient Descent", n_hidden=100):
    file.write("---------------------------------------------------------------------------------------------\n")
    file.write(title + "\n")
    # n_experiment = 100
    n_experiment = 1

    # Create placeholder to accumulate prediction accuracy
    accs = np.zeros(n_experiment)

    for k in range(n_experiment):
        file.write ("Experiment " + str(k) + "\n")
        # Reset model
        model = make_network(n_hidden=n_hidden)

        # Train the model
        model = sgd(model, X_train, y_train, minibatch_size)

        y_pred = np.zeros_like(y_test)

        for i, x in enumerate(X_test):
            # Predict the distribution of label
            _, prob = forward(x, model)
            # Get label by picking the most probable one
            y = np.argmax(prob)
            y_pred[i] = y

        # Compare the predictions with the true labels and take the percentage
        accs[k] = (y_pred == y_test).sum() / y_test.size
        file.write(str(y_pred) + "\n")
        plt.title(title)
        plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, edgecolor='k', s=20)
        plt.savefig("B/images/" + title.replace(" ", "-"))

    file.write('Mean accuracy: {}, std: {}\n'.format(accs.mean(), accs.std()))

def useIris():
    iris = load_iris()
    X = iris.data[0:99]
    y = iris.target[0:99]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20)

    X_train_Sepal = X_train[:, :2]
    X_test_Sepal = X_test[:, :2]

    plotData(X_train, y_train, "data.png")

    doGradientDecent(X_train_Sepal, X_test_Sepal, y_train, y_test, minibatch_size=1, title="Batch Gradient Descent")
    doGradientDecent(X_train_Sepal, X_test_Sepal, y_train, y_test, minibatch_size=len(X_train), title="Stochastic Gradient Descent")
    doGradientDecent(X_train_Sepal, X_test_Sepal, y_train, y_test, minibatch_size=50, title="Mini Batch Gradient Descent")

def useMoons():
    X, y = make_moons(n_samples=10000, random_state=42, noise=0.1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

    plotData(X_train, y_train, "data.png")

    doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size=1, title="Batch Gradient Descent")
    doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size=len(X_train), title="Stochastic Gradient Descent")
    doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size=50, title="Mini Batch Gradient Descent")

def useClassification():
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,random_state=0)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
    plotData(X_train, y_train, "data.png")

    doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size=1, title="Batch Gradient Descent")
    doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size=len(X_train), title="Stochastic Gradient Descent")
    doGradientDecent(X_train, X_test, y_train, y_test, minibatch_size=50, title="Mini Batch Gradient Descent")

if __name__ == "__main__":
    file = open('metricsB.txt', 'w')
    # useMoons()
    useClassification()
