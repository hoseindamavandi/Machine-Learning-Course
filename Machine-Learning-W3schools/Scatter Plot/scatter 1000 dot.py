import matplotlib.pyplot as plt
import numpy

x = numpy.random.normal(5.0,1.0,1000)
y = numpy.random.normal(10.0,2.0,1000)

plt.scatter(x,y)
plt.show()