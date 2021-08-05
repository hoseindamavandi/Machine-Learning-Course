import matplotlib.pyplot as plt
from scipy import stats

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slop, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slop * x + intercept

mymodel = list(map(myfunc,x))

plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()