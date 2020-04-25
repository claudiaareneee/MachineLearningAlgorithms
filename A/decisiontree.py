# pylint: disable=maybe-no-member
from sklearn import tree
import graphviz 

def decisionTree(X_train, y_train, max_depth=0):
    if max_depth is not 0:
        model = tree.DecisionTreeClassifier(max_depth=max_depth)
    else:
        model = tree.DecisionTreeClassifier()
    
    model = model.fit(X_train, y_train)
    return model


def plotDecisionTree(model):
    tree.plot_tree((model)) 

    dot_data = tree.export_graphviz(model, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("A/images/decisiontree") 