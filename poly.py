# This program implements the polynomial regression using Salary based on Years of Experience
# Author: Chris Bagalwa
# 02/11/2022

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Define X and Y training set variable consististing of salary (K Rands) earned over a certain period of time (Years)
X = [[1], [2], [3], [4], [5],[6], [7], [8], [9], [10]]
y = [[4500], [5000], [6000], [8000], [11000],[15000], [20000], [30000], [50000], [70000]]

# Use 20% of the data set for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Train the Linear Regression model and plot a prediction
model = LinearRegression()
model.fit(X_train, y_train)
xx = np.linspace(1, 22, 100)
yy = model.predict(xx.reshape(xx.shape[0], 1))

# Find relation of different degrees used to create Polynomial degrees
for degree in range(2,12,2):
    # Set the degree of the Polynomial Regression model
    poly_featurizer = PolynomialFeatures(degree = degree)

    # This preprocessor transforms an input data matrix into a new data matrix of a given degree
    X_train_poly = poly_featurizer.fit_transform(X_train)
    X_test_poly = poly_featurizer.transform(X_test)

    # Train and test the regressor_quadratic model
    regressor_poly = LinearRegression()
    regressor_poly = regressor_poly.fit(X_train_poly, y_train)
    xx_poly = poly_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the line 
fig, ax = plt.subplots()
ax.plot(X, y, linestyle = '--', label = 'Model on degree: {}'.format(degree))

# Plotting the graph
plt.title('Salary regressed on years of experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary in Rands')
plt.axis([0, 12, 0, 100000])
plt.grid(True)
# Scatter plot for plotting training points
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X, model.predict(X), label = 'Prediction line', color = 'blue')
plt.legend()
plt.show()