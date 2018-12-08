import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# path of data 
path = 'https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)
df.head()

'''Linear Regression'''

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]

#value of the intercept (a)
lm.intercept_
#value of the Slope (b)
lm.coef_

'''Multiple Linear Regression'''
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])

#value of the intercept (a)
lm.intercept_
#value of the Slope (b)
lm.coef_

#Model Evaluation using Visualization
# import the visualization package: seaborn
import seaborn as sns
#Regression Plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

#Residual Plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()

#Multiple Linear
Y_hat = lm.predict(Z)
Y_hat[0:5]

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()

#PlotPOly
def PlotPolly(model,independent_variable,dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
print("done")
x = df['highway-mpg']
y = df['price']
print("done")

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPolly(p,x,y, 'highway-mpg')

np.polyfit(x, y, 3)
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1,x,y, 'Length')

#polynomial transform on multiple features
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]

'''Measures for In-Sample Evaluation'''
from sklearn.metrics import mean_squared_error

#for linear regression
#mean_squared_error(Y_true, Y_predict)
mean_squared_error(df['price'], Yhat)
lm.score(Z, df['price'])

from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
r_squared

mean_squared_error(df['price'], ypipe)
mean_squared_error(df['price'], p(x))