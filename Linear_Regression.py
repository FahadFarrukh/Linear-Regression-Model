import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.impute import SimpleImputer

# Load the dataset


data = pd.read_csv("World Energy Consumption.csv")

# Filter data for Pakistan and after 1947
data_pakistan = data[(data['country'] == 'Pakistan') & (data['year'] >= 1965)]

# Select specific columns of interest
data_pakistan = data_pakistan[['year', 'energy_per_capita']]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_pakistan['energy_per_capita'] = imputer.fit_transform(data_pakistan[['energy_per_capita']])

# Plot the data
plt.scatter(data_pakistan['year'], data_pakistan['energy_per_capita'], color="red")
plt.xlabel("Year")
plt.ylabel("Energy per Capita")
plt.title("Energy per Capita in Pakistan (after 1965)")
plt.show()

# Perform linear regression
regr = linear_model.LinearRegression()

train_x = np.array(data_pakistan['year']).reshape(-1, 1)
train_y = np.array(data_pakistan['energy_per_capita']).reshape(-1, 1)

regr.fit(train_x, train_y)
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)

# Plot the regression line
plt.scatter(data_pakistan['year'], data_pakistan['energy_per_capita'], color="red")
plt.plot(train_x, regr.predict(train_x), '-b')
plt.xlabel("Year")
plt.ylabel("Energy per Capita")
plt.title("Linear Regression: Energy per Capita in Pakistan (after 1965)")
plt.show()