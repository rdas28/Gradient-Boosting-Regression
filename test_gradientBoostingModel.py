import argparse
import numpy as np
import csv
from ..models.GradientBoosting import GradientBoostingRegressor

def normalize_data(X):
    normalized_X = []
    for col in zip(*X):  
        min_val = min(col)
        max_val = max(col)
        if max_val - min_val == 0: 
            normalized_X.append([0.5] * len(col))  
        else:
            normalized_X.append([(x - min_val) / (max_val - min_val) for x in col])
    return list(zip(*normalized_X))  # Transposing back to rows

def test_gradient_boosting_regressor(**kwargs):
    # Loading data from CSV
    file_path = kwargs.pop("file_path")
    data = []
    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print("Unable to locate the test data file. Please check if the provided path is valid.")
        return
    except Exception as e:
        print(f"Error is: {e}")
        return

    # Preparing data for training and testing
    try:
        X = np.array([[float(row[key]) for key in row.keys() if key.startswith('x')] for row in data])
        y = np.array([float(row['y']) for row in data])
    except ValueError as e:
        print(f"Error while parsing the data: {e}")
        return

    # # Normalizing X and y
    X = normalize_data(X)
    y_min = min(y)
    y_max = max(y)
    y = [(yi - y_min) / (y_max - y_min) for yi in y] if y_max - y_min != 0 else [0.5] * len(y)

    # Spliting the data into training and testing sets
    split_index = int(0.8 * len(X))
    if split_index == 0 or split_index == len(X):
        print("Not enough data available to divide into training and testing subsets.")
        return


    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Gradient Boosting's Regressor is being trained
    gbm = GradientBoostingRegressor(**kwargs)

    gbm.fit(X_train, y_train)

    # Predicting on testing data
    y_pred = gbm.predict(X_test)

    # Evaluating the model performance
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    mse = np.sum((y_test - y_pred) ** 2) / len(y_test)
    mae = np.sum(np.abs(y_test - y_pred)) / len(y_test)
    rmse = np.sqrt(mse) 

    print("Performance Metrics:")
    print("R² Score: {:.4f}".format(r2_score))
    print("MSE (Mean Squared Error): {:.4f}".format(mse))
    print("MAE (Mean Absolute Error): {:.4f}".format(mae))
    print("RMSE (Root Mean Squared Error): {:.4f}".format(rmse))

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester for Gradient Boosting Regressor")

    # Model parameters
    parser.add_argument("--file_path", type=str, default="gradientboosting/tests/test_data.csv", help="Path to the input CSV file.")
    parser.add_argument("--no_of_estimators", type=int, default=1000, help="Total number of boosting stages.")
    parser.add_argument("--rate_of_learning", type=float, default=0.2, help="Step size shrinkage factor for updates.")
    parser.add_argument("--max_depth", type=int, default=2, help="The maximum depth permitted for individual trees.")
    parser.add_argument("--sample_minimum_split", type=int, default=2, help="Minimum number of samples required to be at a leaf node.")
    parser.add_argument("--subsample", type=float, default=1.0, help="Proportion of samples to draw for fitting each tree.")
    parser.add_argument("--criterion", type=str, default="friedman_mse", help="Metric to evaluate splits.")
    parser.add_argument("--minimum_samples_per_leaf", type=int, default=1, help="Minimum number of samples to require to be at a leaf node.")
    parser.add_argument("--weight_minimum_leaf_fraction", type=float, default=0.0, help="Smallest weighted fraction of sum total of samples required to be at a leaf.")
    parser.add_argument("--least_impurity_reduce", type=float, default=0.0, help="Minimum impurity decrease required for a node to be split.")
    parser.add_argument("--random_state", type=int, default=None, help="Seed used to generate pseudorandom numbers.")
    parser.add_argument("--how_many_features", type=str, default=None, help="Limit of features to consider for a split at each node.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level for logs and output.")
    parser.add_argument("--greatest_node_of_leaf", type=int, default=None, help="Limit on the total number of leaf nodes.")
    parser.add_argument("--criteria_for_early_stopping", type=float, default=0.1, help="Fraction of the training data to reserve for validation.")
    parser.add_argument("--count_of_max_iteration", type=int, default=None, help="Early stopping maximum iterations without improvement.")
    parser.add_argument("--tol", type=float, default=0.0001, help="Early stopping tolerance requirement.")
    parser.add_argument("--alpha_parameter", type=float, default=0.0, help="Complexity parameter used for pruning.")


    args = parser.parse_args()

    # Convertint arguments to a dictionary and passing to the test function for testing
    test_gradient_boosting_regressor(**vars(args))

