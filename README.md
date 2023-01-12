# Polynomial-Regression
This program implements the polynomial regression using Salary based on Years of Experience
# Description
This program is an implementation of polynomial regression using salary as the dependent variable and years of experience as the independent variable. The program uses the sklearn library to train and test a linear regression model, as well as to create polynomial features of different degrees. The program also uses the matplotlib library to plot the data and the prediction line.

The program starts by importing the necessary libraries: numpy, matplotlib, LinearRegression and PolynomialFeatures classes from the sklearn library. It then defines the X and Y training set variables, consisting of salary and years of experience data respectively. The program then splits the data into a training set and a testing set, using 20% of the data for testing.

The program then trains the linear regression model and plots a prediction line. It then loops through different degrees of polynomial features, and for each degree, it fits and transforms the training data, trains and tests a regressor model, and plots the prediction line.

The program ends by plotting the data points, the prediction line, and the graph title and labels. It also displays the final plot.
