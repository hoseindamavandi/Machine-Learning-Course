import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")
X = df[['Weight','Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(X,y)

predictedCO2 = regr.predict([[2300,1300]])

print(predictedCO2)
