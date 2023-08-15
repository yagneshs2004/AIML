import seaborn as sns
import numpy as np
import pandas as p

a=p.read_csv("diabetes.csv")
c=a.corr()
sns.heatmap(c,cmap='RdYlGn',annot=True)
