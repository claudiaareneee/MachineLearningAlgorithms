# pylint: disable=maybe-no-member
from sklearn import tree
import graphviz 

def decisionTree(iris):
    model = tree.DecisionTreeClassifier()
    model = model.fit(iris.data, iris.target)
    return model


def plotDecisionTree(iris, model):
    tree.plot_tree(model.fit(iris.data, iris.target)) 

    dot_data = tree.export_graphviz(model, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("A/images/decisiontree") 