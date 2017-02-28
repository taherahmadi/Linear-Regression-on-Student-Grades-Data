import math
import numpy as np


# Fits a line that minimizes Residual Sum of Errors
# The line is Yi = B0 + B1*Xi
# feature_vector(Xi) and response_vector(Yi) are numpy arrays
# returns B0, B1, sigma2_hat(estimated variance of epsilon)
# standard errors of B0 and B1, RSS and R2 Metrics
def rss_regressor(feature_vector, response_vector, feature_vector_test, response_vector_test):
    Y_bar = sum(response_vector) / len(response_vector)
    Y_bar_test = sum(response_vector_test) / len(response_vector_test)
    X_bar = sum(feature_vector) / len(feature_vector)
    B1_hat = sum((feature_vector - X_bar) * (response_vector - Y_bar)) / sum((feature_vector - X_bar) ** 2)
    B0_hat = Y_bar - B1_hat * X_bar

    RSS_train = sum((response_vector - (B0_hat + B1_hat * feature_vector)) ** 2)
    RSS_test = sum((response_vector_test - (B0_hat + B1_hat * feature_vector_test)) ** 2)

    sigma2_hat = 1.0 / (len(feature_vector) - 2.0) * RSS_train

    S_x = (1.0 / len(feature_vector)) * sum((feature_vector - X_bar) ** 2)
    standard_error_B1 = math.sqrt(sigma2_hat) / S_x * math.sqrt(len(feature_vector))
    standard_error_B0 = standard_error_B1 * math.sqrt(sum(feature_vector ** 2) / len(feature_vector))

    ESS_train = sum((response_vector - Y_bar) ** 2)
    R2_train = 1.0 - (RSS_train / ESS_train)
    ESS_test = sum((response_vector_test - Y_bar_test) ** 2)
    R2_test = 1.0 - (RSS_test / ESS_test)
    return B0_hat, B1_hat, sigma2_hat, standard_error_B0, standard_error_B1, RSS_train, R2_train, RSS_test, R2_test


# Gets points of fitted line for plotting purpose
def get_points(line, B0, B1):
    points = []
    for i in line:
        points.append(B0 + B1 * i)
    return points

# Likelihood function calculator
def log_likelihood(response_vector, xy):
    N = len(response_vector)
    sigma2 = (1.0 / (N - 2)) * sum((response_vector - xy) ** 2)
    l = -N * (np.log(math.sqrt(sigma2))) - (1.0 / (2 * sigma2)) * sum((response_vector - xy) ** 2)
    return l

# Fits a line that minimizes Residual Sum of Errors
# The line is Yi = B0 + B1*Xi
# feature_vector(Xi) and response_vector(Yi) are numpy arrays
# feature_vector is n*k matrix where k is number of co-variates
# returns B0, B1, sigma2_hat(estimated variance of epsilon)
# standard errors of B0 and B1, RSS and R2 Metrics
def multivariate_rss_regressor(feature_vector, response_vector, feature_vector_test, response_vector_test):
    beta_hat_vector = np.dot(np.dot(np.linalg.inv(np.dot(feature_vector.T, feature_vector)), feature_vector.T),
                             response_vector)
    residual_vector = np.dot(feature_vector, beta_hat_vector) - response_vector
    S = len(beta_hat_vector)

    RSS_train = sum(residual_vector ** 2)
    TSS_train = sum((response_vector - sum(response_vector) / len(response_vector)) ** 2)
    R2_train = 1.0 - (RSS_train / TSS_train)

    log_likelihood2 = log_likelihood(response_vector, np.dot(feature_vector, beta_hat_vector))
    AIC = log_likelihood2 - S

    return AIC, RSS_train, R2_train, beta_hat_vector


# Calculates Leave One Out Cross Validation Error
def LOOCV(features_to_use, train_label):
    RSS_list = []
    for i in range(0, features_to_use.shape[1]):
        temp_data = np.delete(features_to_use.T, i, 0)
        temp_label = np.delete(train_label.reshape(len(train_label), 1), i, 0)
        all_feature_model = multivariate_rss_regressor(temp_data, temp_label, None, None)
        RSS_list.append(all_feature_model[1])
    LOOCV = sum(RSS_list) / len(RSS_list)
    return LOOCV
