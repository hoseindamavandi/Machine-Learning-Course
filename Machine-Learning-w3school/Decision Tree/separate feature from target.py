import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv('shows.csv')
d = {'UK': 0 , 'USA': 1 , 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1 , 'NO':0}
df['Go'] = df['Go'].map(d)

feature = ['Age', 'Experience', 'Rank', 'Nationality']
x = df[feature]
y = df['Go']

#print(df)
print(x)
print(y)