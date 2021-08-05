import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(0.0,5.0,100000)

plt.hist(x,100)
plt.show()