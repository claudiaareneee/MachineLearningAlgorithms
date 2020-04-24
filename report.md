# Machine Learning Algorithms

The goal of this project is to comapre different machine learning algorithms for different problems. Part A of this application focuses on using the Iris dataset to compare classification algorithms. Part B of this project focuses on different implementations of gradient decent.

## Part A - Classification Algorithms

_Two to three pages_

### What you did

Part A of this project was designed to teach about different classification algorithms. Several different classification algorithms were used on the famous Iris dataset to predict the species of an Iris given the sepal width, sepal length, petal width, and petal length. These classification algorithms include the decision tree, k-nearest neighbors, neural networks through backpropagation, and k-means clustering algorithms. These algorithms were implemented through [Scikit-learn](https://scikit-learn.org/stable/index.html).</br>

This project was developed using [Python 3.7.7](https://www.python.org/downloads/release/python-370/) and [Scikit-Learn](https://scikit-learn.org/stable/index.html). Additionally, the machine used to build and run the code is running MacOs Mojave (v10.14.6).</br>

The Iris dataset is comprised of 150 samples, and each sample records the width and length of both the petal and sepal. To visualize these samples, two graphs were created: one comparing petal width and length, the other comparing sepal width and length. The following images show these relationships and the categorization of each data point.</br>

Petal                                                                       |  Sepal
:--------------------------------------------------------------------------:|:--------------------------------------------------------------------------:
![Graph](A/images/Iris-classification-according-to-Petal-measurements.png)  |  ![Graph](A/images/Iris-classification-according-to-Sepal-measurements.png)

On these graphs, it is important to note that datapoints of different categories can overlap. For instance, there are two datapoints at sepal length 5.9 and sepal width 3, but one is categorized as versicolor and the other virginica. Taken without the petal length and width, it is impossible to distinguish these two points into seperate categories. </br>

For each algorithm, the Iris data set was split into test data and training data using the ```train_test_split``` method from SciKit-Learn. Twenty percent of samples were saved for testing data, where the other samples were used for training data. Next the models were fit to the training data, and test samples were passed in to check for accuracy. To better visualize the results of these algorithms, the models were then fit to petal data and sepal data independently and graphed.

### Results

Which classification learning algorithm performed best?

### Lessons Learned

### Digit Data Comparison

## Part B - Gradient Decent

_Two pages_

### What you did

### Results

Compare the training time and final performance of the neural networks trained using these three variants of gradient descent and report your results using at least one graph.

### Analysis

Include a description of the neural network you trained: how many hidden layers and hidden units, how many input and output nodes, and the details of the learning algorithm (learning rate, momentum).