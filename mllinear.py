# -- coding: utf-8 --
"""
#AbubakarAsif FA20-BCE-013
#abubakarasif3111@gmail.com
#https://github.com/Abubakar3111
#https://www.linkedin.com/in/abubakar-asif-b3b94021a/
print("\nAbubakar Asif")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('marks.csv')
print(data.shape)
print(data)
x1 = data['Quiz'].values
x2 = data['Assg'].values
x3 = data['Mid'].values
Y = data['Final'].values
m = len(x1)
x1 = x1.reshape(m)
x2 = x2.reshape(m)
x3 = x3.reshape(m)
ax = plt.axes(projection ='3d')
ax.scatter(x1, x2, Y)
x0=np.ones(m)
X=np.array([x0,x1,x1**2,x2,x2**2,x3,x3**2]).T
print(X)
reg=LinearRegression()
reg.fit(X,Y)
print("\nData Score:",reg.score(X,Y))
h_theta=reg.predict(X)
print("\nAbubakar Asif  fa20-bce-013" )
quiz=78
Assg=83
Mid=77
presal=reg.predict( [[1,quiz,quiz**2,Assg,Assg**2,Mid,Mid**2]])
print("\nMy Output\n")
print(presal)
print("\nActual Output:160")