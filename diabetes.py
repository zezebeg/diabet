from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from six import StringIO

veri = pd.read_csv("diabet.csv")

abc = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = veri[abc]
y = veri.Outcome


clf = DecisionTreeClassifier(random_state=0)

model = clf.fit(X, y)

dot_data = StringIO()

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=abc,  class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
