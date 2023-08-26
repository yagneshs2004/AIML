import pandas as p
from sklearn.datasets import load_iris
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression


d=load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(d.data, d.target, test_size=0.2)
model3=LogisticRegression()
model3.fit(xtrain,ytrain)
predicted=model3.predict(xtest)

actual=ytest
con=metrics.confusion_matrix(predicted,actual)
cmdisplay=metrics.ConfusionMatrixDisplay(confusion_matrix=con)
cmdisplay.plot()
print(model3.score(xtest,ytest))