import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as p
from sklearn.linear_model import LogisticRegression

a=p.read_csv("diabetes.csv")
cor=a.corr()
sns.heatmap(cor,annot=True)
x=a.drop(columns=['Outcome'])
y=a['Outcome']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
model1=LogisticRegression()
model1.fit(xtrain,ytrain)
predictions=model1.predict(xtest)
actual=ytest
confusion_matrix = metrics.confusion_matrix(actual, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True, False])
cm_display.plot()
score=accuracy_score(actual,predictions)
print(score)
