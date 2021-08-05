import numpy as np
from sklearn.metrics import r2_score
np.random.seed(2)

x = np.random.normal(3,1,100)
y = np.random.normal(150,40,100)/x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = np.poly1d(np.polyfit(train_x,train_y,4))
r2 = r2_score(train_y,mymodel(train_x))
print(r2)