import pandas
from sklearn import linear_model

df = pandas.read_csv('cars.csv')
X = df[['Weight','Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(X,y)
PredicatedCO2 = regr.predict([[3300,1300]])
print(PredicatedCO2)