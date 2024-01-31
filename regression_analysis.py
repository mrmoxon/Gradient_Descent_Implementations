import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import datasets

import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation
import random

learning_rate = 0.001
epochs = 1000

data = pd.read_csv('data/london-borough-profiles-jan2018.csv', encoding='latin1')

x = data['Male life expectancy, (2012-14)'] = pd.to_numeric(data['Male life expectancy, (2012-14)'], errors='coerce')
y = data['Female life expectancy, (2012-14)'] = pd.to_numeric(data['Female life expectancy, (2012-14)'], errors='coerce')

# ### Partitioning the Data

print(x)

# For future purposes, we will reshape the data to be used in the regression model
x_reshaped = x.to_numpy().reshape(-1, 1)

# First splitting to create training and test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_reshaped, y, test_size=0.1)
# Further split training set to create validation set
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.1)

# Plot training data with blue color and circle marker
plt.scatter(x_train, y_train, color='blue', marker='o', label='Train')
# Plot test data with red color and x marker
plt.scatter(x_test, y_test, color='red', marker='x', label='Test')

plt.xlabel('Male life expectancy, (2012-14)')
plt.ylabel('Female life expectancy, (2012-14)')
plt.legend()
plt.show()

# ### Generate Synthetic Dataset

# 2x the size of the previous observation set
xs, ys, ps = datasets.make_regression(n_samples =70, n_features = 1, n_informative = 1, noise = 35, coef = True, random_state=112)
# First splitting to create training and test sets
xs_train, xs_test, ys_train, ys_test = model_selection.train_test_split(xs, ys, test_size=0.1)
# Further split training set to create validation set
xs_train, xs_val, ys_train, ys_val = model_selection.train_test_split(xs_train, ys_train, test_size=0.1)

plt.scatter(xs_train, ys_train, color='blue', marker='o', label='Train')
plt.scatter(xs_test, ys_test, color='red', marker='x', label='Test')
plt.xlabel('Male life expectancy, (2012-14)')
plt.ylabel('Female life expectancy, (2012-14)')
plt.show()

# ### Linear Regression on Data (I): Gradient Descent

print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_val: ', x_val.shape)
print('y_val: ', y_val.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Impute missing data in x_train, x_val, x_test
x_train = imputer.fit_transform(x_train)

x_val = imputer.transform(x_val)
x_test = imputer.transform(x_test)

# Impute missing data in y_train, y_val, y_test
# Note: Imputing target variable requires careful consideration and understanding of the data
y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val = imputer.transform(y_val.values.reshape(-1, 1)).ravel()
y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

print('x_train = \n', x_train)
print('y_train = \n', y_train)
print('x_val = \n', x_val)
print('y_val = \n', y_val)
print('x_test = \n', x_test)
print('y_test = \n', y_test)

# We will:
# 
# 1. Initialise the weights randomly.
# 
# 2. Compute the gradient with respect to each weight.
# 
# 3. Update the weights by subtracting a fraction of the gradient from them.

def initialize_and_predict(X, initial_weights=None):
    """
    Initializes weights if not provided and calculates predictions.

    Args:
    X (np.array): Feature matrix.
    initial_weights (np.array): Initial weights for the model.

    Returns:
    np.array: Predicted values.
    """
    if initial_weights is None:
        # Initialize weights to zeros or small random values
        initial_weights = np.zeros(X.shape[1])

    predictions = X.dot(initial_weights)
    return predictions, initial_weights

initial_predictions, initial_weights = initialize_and_predict(x_train)

# # Compute Errors

#--
# compute_error()
# This function computes the sum of squared errors for the model.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# output:
#  error (scalar)
#--
def compute_error( M, x, w, y ):
    error = 0
    y_hat = [0 for i in range( M )]
    for j in range( M ):
        y_hat[j] = w[0] + w[1] * x[j]
        error = error + math.pow(( y[j] - y_hat[j] ), 2 )
    error = error / M
    return( error )

#--
# compute_r2()
# This function computes R^2 for the model.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# output:
#  r2 (scalar)
#--
def compute_r2( M, x, w, y ):
    u = 0
    v = 0
    y_hat = [0 for i in range( M )]
    y_mean = np.mean( y )
    for j in range( M ):
        y_hat[j] = w[0] + w[1] * x[j]
        u = u + math.pow(( y[j] - y_hat[j] ), 2 )
        v = v + math.pow(( y[j] - y_mean ), 2 )
    r2 = 1.0 - ( u / v )
    return( r2 )

# # Run Gradient Descent and Build GIF

# Normalize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
x_test_scaled = scaler.transform(x_test).flatten()

# Initialize weight and bias (used) globally
w = random.random()
b = random.random()

# # Generate a Standard Score

# <ins> Mean Squared Error </ins>
# 
# (An ideal score would be 0)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate the final error and R-squared on the test set
y_pred_test = w * x_test_scaled + b
test_error = compute_error(len(y_test), x_test_scaled, [b, w], y_test)

sklearn_mse_test = mean_squared_error(y_test, y_pred_test)

def manual_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

manual_mse_test = manual_mse(y_test, y_pred_test)
print(f'MSE (Test Set): {manual_mse_test}')

# <ins> Sum of Squared Errors </ins>
# 
# (An ideal score would be 0)
# 
# (Same as MSE for one datapoint)

def manual_sse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

manual_sse_test = manual_sse(y_test, y_pred_test)
print(f'MSE (Test Set): {manual_mse_test}')

# <ins> Root Mean Squared Error </ins>
# 
# (An ideal score would be 0)

sklearn_rmse_test = np.sqrt(sklearn_mse_test)

def manual_rmse(y_true, y_pred):
    return np.sqrt(manual_mse(y_true, y_pred))

manual_rmse_test = manual_rmse(y_test, y_pred_test)
print(f'RMSE (Test Set): {manual_rmse_test}')

# <ins> Mean of Absolute Differences </ins>
# 
# (An ideal score would be 0)

sklearn_mad_test = mean_absolute_error(y_test, y_pred_test)

def manual_mad(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

manual_mad_test = manual_mad(y_test, y_pred_test)
print(f'MAD (Test Set): {manual_mad_test}')

# <ins> R-Squared </ins>
# 
# (An ideal model would be 1)

sklearn_r2_test = r2_score(y_test, y_pred_test)

def manual_r2(y_true, y_pred):
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    ssr = np.sum((y_true - y_pred) ** 2)
    return 1 - (ssr/sst)

manual_r2_test = manual_r2(y_test, y_pred_test)
print(f'R-squared (Test Set): {manual_r2_test}')

# <ins> Define Learning Rate and Epochs </ins>
# 
# (more sophisticated would use convergence criteria as epochs)

# Set up the figure
fig, ax = plt.subplots()
ax.scatter(x_train_scaled, y_train, color='blue', label='Data Points')
line, = ax.plot(x_train_scaled, x_train_scaled * w + b, color='red', label='Regression Line')
plt.legend()

# Set initial y-axis limits
ax.set_ylim(min(y_train) - 1, max(y_train) + 1)

# Initialize lists to store error metrics
mse_values = []
rmse_values = []
mad_values = []
r2_values = []

# Update function for animation
def update(epoch):
    global w, b # Need to modify global copy of w and b (weights and bias)

    # Predict the output using the current weights and bias and calculate the error
    y_pred = w * x_train_scaled + b
    error = y_train - y_pred
    mse = np.mean(error**2)

    # Calculate the gradients for weights and bias
    dw = -2 * np.dot(x_train_scaled.T, error) / len(x_train_scaled)
    db = -2 * np.sum(error) / len(x_train_scaled)

    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

    # Update the regression line
    line.set_ydata(y_pred)

    # Calculate and append the error metrics
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    mad = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    mse_values.append(mse)
    rmse_values.append(rmse)
    mad_values.append(mad)
    r2_values.append(r2)

    
    # Dynamically adjust the minimum y-value of the plot
    current_min_y = min(y_train.min(), y_pred.min()) - 1
    ax.set_ylim(current_min_y, ax.get_ylim()[1])

    # Update the plot
    # line.set_ydata(x_train_scaled * w + b)
    ax.set_title(f'Epoch {epoch} - MSE: {mse:.4f}')
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, epochs), interval=50)

# Save to GIF (this requires ffmpeg or imagemagick to be installed)
ani.save(f'gifs/linear_regression_L{learning_rate}_E{epochs}.gif', writer='imagemagick')

plt.show()

# <ins> Plotting the Respective Errors </ins>
# 
# We are looking for convergence on zero from positive values for all but R^2.
# 
# The scientific notation equates to scale factor of graph. 
# Plot the error metrics with adjusted axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Error Metrics Over {epochs} Epochs with Learning Rate {learning_rate}', fontsize=16)

# For MSE
axes[0, 0].plot(mse_values, label='MSE')
axes[0, 0].set_title('Mean Squared Error')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE')
# axes[0, 0].ticklabel_format(style='plain', axis='x')  # Disable scientific notation

# For RMSE
axes[0, 1].plot(rmse_values, label='RMSE', color='orange')
axes[0, 1].set_title('Root Mean Squared Error')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('RMSE')
# axes[0, 1].set_ylim([min(rmse_values), max(rmse_values)])  # Adjust y-axis range

# For MAD
axes[1, 0].plot(mad_values, label='MAD', color='green')
axes[1, 0].set_title('Mean Absolute Deviation')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAD')
# axes[1, 0].set_ylim([min(mad_values), max(mad_values)])  # Adjust y-axis range

# For R-squared
axes[1, 1].plot(r2_values, label='R-squared', color='red')
axes[1, 1].set_title('R-squared Score')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('R-squared')
# axes[1, 1].set_ylim([min(r2_values), max(r2_values)])  # Adjust y-axis range

plt.tight_layout()

# Create and save the animation
plt.savefig(f'graphs/error_graphs_L{learning_rate}_E{epochs}.png', dpi=300)

plt.show()

# Make predictions with the trained model
y_pred_train = w * x_train_scaled + b
y_pred_test = w * x_test_scaled + b

# Visualization
metrics = ['SSE', 'MSE', 'RMSE', 'MAD', 'R-squared']
manual_scores = [manual_sse_test, manual_mse_test, manual_rmse_test, manual_mad_test, manual_r2_test]
sklearn_scores = [manual_sse_test, sklearn_mse_test, sklearn_rmse_test, sklearn_mad_test, sklearn_r2_test]  # SSE is the same

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(f'Comparison of Regression Errors Over {epochs} Epochs with Learning Rate {learning_rate}', fontsize=16)

rects1 = ax.bar(x - width/2, manual_scores, width, label='Manual')
rects2 = ax.bar(x + width/2, sklearn_scores, width, label='sklearn')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Comparison of Regression Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig(f'graphs/compare_errors_L{learning_rate}_E{epochs}.png', dpi=300)

plt.show()

# # 2. Run for synthetic data: it should take 10k iterations apparently?
# # 3. Show the convergence gif for a higher learning rate. Should converge within a few iterations.