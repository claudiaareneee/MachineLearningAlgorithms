# https://towardsdatascience.com/an-introduction-to-gradient-descent-c9cca5739307

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from batch import Batch_GD
from minibatch import Minibatch_GD
from stochastic import Stochastic_GD

# Load data
iris = datasets.load_iris()
X=iris.data[0:99,:2]
y=iris.target[0:99]
# Plot the training points
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Run a function
Result_BatchGD=Batch_GD(Learning_Rate=0.01,num_iterations=100000,X=X,y=y)

# Run a function
Result_Stoc_GD=Stochastic_GD(Learning_Rate=0.01,num_iterations=2000,X=X,y=y)

# Run a function
Result_MiniGD=Minibatch_GD(Learning_Rate=0.01,num_iterations=100000,X=X,y=y,Minibatch=50)

# Plot linear classification
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
line_B_GD=mlines.Line2D([0,7],[-0.5527,4.1577],color='red')
line_Mini_GD=mlines.Line2D([0,7],[-0.56185,4.1674],color='blue')
line_Sto_GD=mlines.Line2D([0,7],[-0.5488,4.1828],color='green')
ax.add_line(line_B_GD)
ax.add_line(line_Mini_GD)
ax.add_line(line_Sto_GD)
ax.set_xlabel('Sepal length')
plt.show()