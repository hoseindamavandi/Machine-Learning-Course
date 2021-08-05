import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
x = np.random.normal(3,1,100)
y = np.random.normal(150,40,100)/x

plt.scatter(x,y)
plt.show()