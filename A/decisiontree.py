# pylint: disable=maybe-no-member
# sources: https://medium.com/@haydar_ai/learning-data-science-day-21-decision-tree-on-iris-dataset-267f3219a7fa
from sklearn import tree
import graphviz 

def decisionTree(X_train, y_train, max_depth=0):
    if max_depth is not 0:
        model = tree.DecisionTreeClassifier(max_depth=max_depth)
    else:
        model = tree.DecisionTreeClassifier()
    
    model = model.fit(X_train, y_train)
    return model


def plotDecisionTree(model, name):
    tree.plot_tree((model)) 

    dot_data = tree.export_graphviz(model, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("A/images/" + name.replace(' ', '')) 