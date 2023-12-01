#!/usr/bin/python3
# Homework 5 Code
import matplotlib
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', 'GTK3Agg', etc.


def split_data(first, second):
    # Step a1: Load the Data
    train_data = np.loadtxt('zip.train')
    test_data = np.loadtxt('zip.test')

    # Step a2: Filter the Data for Binary Classification
    # Let's say we're doing 1 vs. 3 first
    train_data_First_VS_Second = train_data[(train_data[:, 0] == first) | (train_data[:, 0] == second)]
    test_data_First_VS_Second = test_data[(test_data[:, 0] == first) | (test_data[:, 0] == second)]

    # Step a3: Split the Data into Features and Labels
    X_train_First_VS_Second = train_data_First_VS_Second[:, 1:]  # all columns except the first
    y_train_First_VS_Second = train_data_First_VS_Second[:, 0]  # first column is the label

    X_test_First_VS_Second = test_data_First_VS_Second[:, 1:]
    y_test_First_VS_Second = test_data_First_VS_Second[:, 0]

    # Convert labels to binary (-1 for digit `second`, +1 for digit `first`)
    y_train_First_VS_Second = np.where(y_train_First_VS_Second == first, 1, -1)
    y_test_First_VS_Second = np.where(y_test_First_VS_Second == first, 1, -1)

    return X_train_First_VS_Second, y_train_First_VS_Second, X_test_First_VS_Second, y_test_First_VS_Second
def calculate_adaboost_error(stumps, stump_weights, X, y):
    # Compute the AdaBoost ensemble prediction
    stump_predictions = np.array([stump.predict(X) for stump in stumps])
    ensemble_prediction = np.sign(np.dot(stump_weights, stump_predictions))

    # Compute the error as the fraction of misclassified examples
    return np.mean(ensemble_prediction != y)


def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    m = X_train.shape[0]  # Number of training samples
    D = np.ones(m) / m  # Initialize weights

    # The weak learners and their alpha values will be stored here
    stumps = []
    stump_weights = []

    # Placeholder for the errors at each iteration
    train_error = np.zeros(n_trees)
    test_error = np.zeros(n_trees)

    for t in range(n_trees):
        # Train the stump
        stump = DecisionTreeClassifier(max_depth=1)  # Decision stump is a tree of depth 1
        stump.fit(X_train, y_train, sample_weight=D)
        stump_pred = stump.predict(X_train)

        # Calculate the error and alpha
        err = np.sum(D * (stump_pred != y_train))
        alpha = 0.5 * np.log((1 - err) / err)

        # Store the stump and its alpha
        stumps.append(stump)
        stump_weights.append(alpha)

        # Update the weights D
        D *= np.exp(-alpha * y_train * stump_pred)
        D /= np.sum(D)  # Normalize the weights

        # Calculate and store the errors
        train_error[t] = calculate_adaboost_error(stumps, stump_weights, X_train, y_train)
        test_error[t] = calculate_adaboost_error(stumps, stump_weights, X_test, y_test)

    # TODO: Remove Print the final test error for debugging
    print(f"Final train error: {test_error}")
    print(f"Final test error: {test_error}")

    return train_error, test_error


def main_hw5():

    # Split the training data for (1 VS 3) and (3 VS 5)
    X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3 = split_data(1, 3)
    X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5 = split_data(3, 5)

    num_trees = 200

    # Train and get error rates for 1 vs 3
    train_error_1v3, test_error_1v3 = adaboost_trees(X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3, num_trees)
    # Train and get error rates for 3 vs 5
    train_error_3v5, test_error_3v5 = adaboost_trees(X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5, num_trees)

    # Plotting for 1 vs 3
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_trees + 1), train_error_1v3, label='Train Error (1 vs 3)')
    plt.plot(range(1, num_trees + 1), test_error_1v3, label='Test Error (1 vs 3)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost Error Rates for 1 vs 3')
    plt.legend()
    plt.show()

    # Plotting for 3 vs 5
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_trees + 1), train_error_3v5, label='Train Error (3 vs 5)')
    plt.plot(range(1, num_trees + 1), test_error_3v5, label='Test Error (3 vs 5)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost Error Rates for 3 vs 5')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main_hw5()
