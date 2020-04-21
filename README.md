# MachineLearningAlgorithms

This is a project created for CSE 4633/6633 Introduction to Artificial Intelligence. The objective is to compare different machine learning algorithms and determine their effectiveness.

## Instructions

### Quick information

* Course: CSE 4633/6633 Introduction to Artificial Intelligence
* Due date: Tuesday, April 28, by midnight
* Type: Programming Assignment
* (Instead of this assignment, you can do a machine learning project of your own choice, subject to your instructor’s approval of the topic)

### Part A

In the first part of the assignment, you will familiarize yourself with some of the machine learning resources available in Python, and compare the performance of three classification algorithms. Instead of implementing the algorithms yourself, you will use a Python library.

#### Scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/index.html) is a free software machine learning library for the Python programming language that features various learning algorithms designed to be used with the Python numerical and scientific libraries NumPy and SciPy. Several tutorials on how to use scikit-learn can be found online, including one found [here](https://www.datacamp.com/community/tutorials/machine-learning-python).

#### Iris Dataset

Scikit-learn includes several standard datasets. Among them is the Iris dataset, a classic “toy” problem that is widely-used in the machine learning literature. The dataset is very small, with only a 150 samples. There are 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Many tutorials about the Iris dataset can be found online, including the one found at [opensource.com](https://opensource.com/article/18/9/how-use-scikit-learn-data-science-projects).

#### Machine learning algorithms

Scikit-learn includes implementations of several learning algorithms. You assignment is to apply three supervised learning algorithms to the Iris dataset: the decision tree classification algorithm with the information gain criterion, the backpropagation algorithm, and the k-nearest neighbors classification algorithm. In addition, you should apply an unsupervised learning algorithm to the dataset: k-means clustering.</br>

Many tutorials on how to use these algorithms can be found online, including the following:

* For all algorithms: [Brainscribble](http://stephanie-w.github.io/brainscribble/classification-algorithms-on-iris-dataset.html) and [CodeBagNg](https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342)
* For decision tree learning: [Haydar AI](https://medium.com/@haydar_ai/learning-data-science-day-21-decision-tree-on-iris-dataset-267f3219a7fa)
* For backpropagation: [Stackabuse](https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/)
* For k-nearest neighbors: [Stackabuse](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/) and [GeeksForGeeks](https://www.geeksforgeeks.org/introduction-machine-learning-using-python/)
* For k-means clustering: [Constantgeeks](http://constantgeeks.com/playing-with-iris-data-kmeans-clustering-in-python/)

#### Part A Deliverables

Turn in any Python code you wrote, which will use the scikit-learn library, and a two-to-three page report that describes what you did, reports your results, and describes what you learned. Which classification learning algorithm performed best? (Since k-means clustering is not a classification algorithm, its performance cannot be compared to the other algorithms.)

#### Optional extra credit

This is the “A” option: Compare the same three classification algorithms from part A of the assignment on the digits dataset that is also included in the scikit-learn library. For this dataset, the task is to predict, given an image, which of ten possible digits it represents. Many tutorials for the digits dataset can be found online, including this [one](https://www.datacamp.com/community/tutorials/machine-learning-python).

### Part B

For the rest of the assignment (which is required), compare the performance of three variants of gradient descent used in a backpropagation algorithm for training a multi-layer feedforward neural network: batch gradient descent, stochastic gradient descent, and mini-batch gradient descent. 

* *Batch gradient descent* updates the network weights after computing the gradient of error with respect to the entire training set.
* *Stochastic gradient descent* updates the network weights after computing the gradient of error with respect to a single training example.
* *Mini-batch gradient descent* updates the network weights after computing the gradient of error with respect to a subset of the training set, where the size of the subset can be varied to optimize performance.

The following articles describe these three forms of gradient descent:

* [Machine Larning Mastery](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/ )
* [Adventures in Machine Learning](https://adventuresinmachinelearning.com/stochastic-gradient-descent/)

#### Code

You can use or modify any Python code you find online, as long as you cite the source. However, do not use an implementation of the algorithm from a library or toolbox; it should be code that you can read and modify, and we strongly recommend that you use code that is as simple and clear as possible, so that you can easily understand it.

#### Datasets

You can find many widely-used machine learning datasets in the Machine Learning Repository at [UC/Irvine](https://archive.ics.uci.edu/ml/index.php). This repository includes the Iris classification dataset, for example, among hundreds of other problems. You can use any dataset you want for your experiments.

#### Part B Deliverables

Compare the training time and final performance of the neural networks trained using these three variants of gradient descent and report your results using at least one graph. Besides a copy of the Python code used for your experiments, turn in a two-page report that describes what you’ve done, your results and analysis. Include a description of the neural network you trained: how many hidden layers and hidden units, how many input and output nodes, and the details of the learning algorithm (learning rate, momentum).
