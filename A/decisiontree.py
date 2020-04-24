# pylint: disable=maybe-no-member
from sklearn import tree
import graphviz 

def decisionTree(iris, max_depth=0):
    if max_depth is not 0:
        model = tree.DecisionTreeClassifier(max_depth=max_depth)
    else:
        model = tree.DecisionTreeClassifier()
    
    model = model.fit(iris.data, iris.target)
    return model


def plotDecisionTree(iris, model):
    tree.plot_tree(model.fit(iris.data, iris.target)) 

    dot_data = tree.export_graphviz(model, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("A/images/decisiontree") 