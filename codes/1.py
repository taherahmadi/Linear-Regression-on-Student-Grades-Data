# Author Taher Ahmadi

import read_data
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import datasets, linear_model
import numpy as np
import random
import methods

data_set, labels = read_data.read('./data_set/Dataset1.csv', 8)

# Split train and test data
train_data = data_set[:400]
train_labels = labels[:400]
test_data = data_set[400:]
test_labels = labels[400:]

print('Train data size:', len(train_data))
print('Test data size:', len(test_data))

# a) Scatter plot each feature vs label
fig1 = plt.figure('a')
gs = gridspec.GridSpec(3, 3)
counter = 0
for i in range(0, 3):
    for j in range(0, 3):
        counter += 1
        if counter == 9:
            break
        ax_temp = fig1.add_subplot(gs[i, j])
        ax_temp.scatter(train_data.get(counter - 1), train_labels,
                        s=10, color='r', alpha=.4)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# b) Finding Simple Linear Regression models for each feature with RSS metric
linear_regressions = []
stats = []
for i in range(0, 8):
    # Create linear regression object
    # regr = linear_model.LinearRegression()
    # fit linear regression on data
    linear_regressions.append(
        linear_model.LinearRegression().fit(train_data.get(i).values.reshape(-1,1), train_labels.values.reshape(-1,1)))
    stats.append(
        methods.rss_regressor(train_data.get(i).values, train_labels,
                              test_data.get(i), test_labels))


# Scatter plot + linear regression on each feature vs label
fig2 = plt.figure('b')
gs = gridspec.GridSpec(3, 3)
counter = 0
for i in range(0, 3):
    for j in range(0, 3):
        counter += 1
        if counter == 9:
            break
        ax_temp = fig2.add_subplot(gs[i, j])
        ax_temp.scatter(train_data.get(counter - 1), train_labels,
                        s=10, color='r', alpha=.4)
        ax_temp.plot(train_data.get(counter - 1).values.reshape(-1,1),
                     linear_regressions[counter - 1].predict(train_data.get(counter - 1).values.reshape(-1,1)),
                     color='blue', linewidth=3)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# Reporting Linear Regression Characteristics for train and test Data
for i in range(0, len(linear_regressions)):
    regression_temp = stats[i]
    b0_hat = regression_temp[0]
    b1_hat = regression_temp[1]
    estimated_epsilon = regression_temp[2]
    standard_error_b0 = regression_temp[3]
    standard_error_b1 = regression_temp[4]
    RSS_train = regression_temp[5]
    R2_train = regression_temp[6]
    RSS_test = regression_temp[7]
    R2_test = regression_temp[8]
    print('Simple Linear Regression with Feature' + str(i + 1) +
          '\nEstimated (Beta0, Beta1): (' + str(b0_hat) + ', ' + str(b1_hat) + ')\n' +
          'Standard Error of Beta0 and Beta1: (' + str(standard_error_b0) + ', ' + str(standard_error_b1) +
          ')\nEstimated Variance of Epsilon: ' + str(estimated_epsilon) + '\n' +
          'RSS_train: ' + str(RSS_train) + str('\n') +
          'R2_train: ' + str(R2_train) + str('\n') +
          'RSS_test: ' + str(RSS_test) + str('\n') +
          'R2_test: ' + str(R2_test) + str('\n'))

# using Feature4(as the best feature) and adding features, then checking AIC, RSS and R2
current_AIC = methods.log_likelihood(train_labels, train_data.get(3)) - 1
print('Current AIC only using Feature4 : ' + str(current_AIC))
used_features_indexes = [3]
features_to_use = train_data.get(3).reshape(1, len(train_data.get(3)))
test_features_to_use = test_data.get(3).reshape(1, len(test_data.get(3)))
improving = True
while improving:
    improvements = {}
    features_to_use_temp = []
    test_features_to_use_temp = []
    for i in range(0, 8):
        if i in used_features_indexes:
            continue
        else:
            features_to_use_temp = np.concatenate((features_to_use, train_data.get(i).
                                                   reshape(1, len(train_data.get(i)))),
                                                  axis=0).T
            test_features_to_use_temp = np.concatenate((test_features_to_use, test_data.get(i).
                                                        reshape(1,
                                                                len(test_data.get(i)))), axis=0).T
            multiple_regression_result = methods.multivariate_rss_regressor(features_to_use_temp, train_labels,
                                                             test_features_to_use_temp,
                                                             test_labels)
            print('AIC after adding Feature' + str(i + 1) + ' :' + str(multiple_regression_result[0]))
            improvements[i] = (multiple_regression_result[0] - current_AIC)
    if len(improvements) > 0 and improvements[max(improvements)] > 0:
        best_index = max(improvements.keys(), key=(lambda k: improvements[k]))
        used_features_indexes.append(best_index)
        features_to_use = np.concatenate((features_to_use, train_data.get(best_index).
                                          reshape(1, len(train_data.get(best_index)))),
                                         axis=0)
        print_temp = 'Model Uses Feature '
        for j in range(0, len(used_features_indexes)):
            print_temp = print_temp + '(' + str(used_features_indexes[j] + 1) + ') '
        print(print_temp)
        multiple_regression_result = \
            methods.multivariate_rss_regressor(features_to_use.T, train_labels, None, None)
        print('RSS: ' + str(multiple_regression_result[1]) + ' and R2: ' + str(multiple_regression_result[2]) + '\n')
        improving = True
    else:
        improving = False


# Evaluating the LOOCV metric for different model
LOOCV = methods.LOOCV(features_to_use, train_labels)
print('Leave One Out Cross Validation Risk for ' + str(features_to_use.shape[0]) + ' Features is: ' +
      str(LOOCV))
temp_features = features_to_use
loocvs = [LOOCV]

# Performing the backward method
for i in range(0, 7):
    temp_features = np.delete(temp_features, 0, 0)
    LOOCV = methods.LOOCV(temp_features, train_labels)
    loocvs.append(LOOCV)
    print('Leave One Out Cross Validation Risk for ' + str(temp_features.shape[0]) + ' Features is: ' +
          str(LOOCV))
plt.plot(np.linspace(1, 8, 8), loocvs)
plt.xlabel('Number of Used Features')
plt.ylabel('RSS')
plt.show()

# Testing Full-Feature model with different number of train data
features_count = [50, 100, 250, 300, 350, 400]
RSS = []
for i in features_count:
    indices = random.sample(range(0, 400), i)
    temp_data = features_to_use.T[indices].reshape(i, 8)
    temp_label = train_labels[indices]
    RSS.append(methods.multivariate_rss_regressor(
        temp_data, temp_label, None, None)[1])
plt.plot(features_count, RSS)
plt.xlabel('Number of Train')
plt.ylabel('Train RSS')

plt.show()
print(RSS)