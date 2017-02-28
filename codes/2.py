# Author Taher Ahmadi

import read_data
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pandas as pd


data_set_1, label_1 = read_data.read('./data_set/Dataset2.csv', 6)

# Split train and test data
train_data_1 = data_set_1[:200]
train_label_1 = label_1[:200]
test_data_1 = data_set_1[200:]
test_label_1 = label_1[200:]

print('Train data size:', len(train_data_1))
print('Test data size:', len(test_data_1))

# Scatter plot each feature vs label
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
counter = 0
for i in range(0, 2):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.scatter(train_data_1.get(counter - 1), train_label_1)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# Filling missing values with Gaussian Noise, N(mean_of_row, 1)
for i in range(0, train_data_1.shape[0]):
    row = train_data_1.values[i]
    for j in range(0, len(row)):
        if row[j] == 0:
            row[j] = abs(np.random.normal(sum(row) / len(row), scale=1))
# Filling missing values of test data with the same way
for i in range(0, test_data_1.shape[0]):
    test_row = test_data_1.values[i]
    if test_row[j] == 0:
        test_row[j] = abs(np.random.normal(sum(test_row) / len(test_row), scale=1))

# Scatter plot each feature vs label after filling missing values
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
counter = 0
for i in range(0, 2):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.scatter(train_data_1.get(counter - 1), train_label_1)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# Using Lasso Regression on Data
# Optimization Objective: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
alpha = 0.000001
lasso_model = Lasso(alpha=alpha)
lasso_line = lasso_model.fit(train_data_1, train_label_1)
title = 'Alpha = ' + str(alpha) + '\nRed: Lasso Prediction, Blue: Real Values'
plt.plot(np.linspace(0, 40, 40), lasso_line.predict(test_data_1), 'r')
plt.plot(np.linspace(0, 40, 40), test_label_1)
plt.title(title)
plt.show()

# Testing for different alphas
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
counter = 0
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
for i in range(0, 2):
    for j in range(0, 3):
        lasso_model = Lasso(alpha=alphas[counter])
        lasso_line = lasso_model.fit(train_data_1, train_label_1)
        predictions = lasso_line.predict(test_data_1)
        if counter == 0:
            RSS = sum((test_label_1 - predictions) ** 2)
            ESS = sum((test_label_1 - (sum(test_label_1) / len(test_label_1))) ** 2)
            R2 = 1 - (RSS / ESS)
            print('RSS: ' + str(RSS) + ', R2: ' + str(R2))
        title = 'Alpha = ' + str(alphas[counter])
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.plot(np.linspace(0, 40, 40), predictions, 'r')
        ax_temp.plot(np.linspace(0, 40, 40), test_label_1)
        ax_temp.title.set_text(title)
        counter += 1
plt.show()

# Predicting the final submission file with the best model
# Filling missing values with Gaussian Noise, N(mean_of_row, 1)
final_data = pd.read_csv('./data_set/Dataset2_Unlabeled.csv', header=None)
for i in range(0, final_data.shape[0]):
    row = final_data.values[i]
    for j in range(0, len(row)):
        if row[j] == 0:
            row[j] = abs(np.random.normal(sum(row) / len(row), scale=1))
lasso_model = Lasso(alpha=0.0001)
lasso_line = lasso_model.fit(train_data_1, train_label_1)
predictions = lasso_line.predict(final_data)
predictions.tofile('./data_set/predictions.csv', sep=',\n')
