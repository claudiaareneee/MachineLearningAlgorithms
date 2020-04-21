# pylint: disable=maybe-no-member
from sklearn import datasets
from sklearn import tree
import graphviz 

iris = datasets.load_iris()

print(iris.data)
print(iris.target)
print(iris.target_names)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

tree.plot_tree(clf.fit(iris.data, iris.target)) 

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 