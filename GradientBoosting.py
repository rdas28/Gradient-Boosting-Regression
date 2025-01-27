import numpy as np
from ..models.DecisionTree import DecisionTree


class GradientBoostingRegressor:
    # Setting up a Gradient Boosting Regressor with key configuration parameters.
    def __init__(self, no_of_estimators=1000, rate_of_learning=0.2, max_depth=2, sample_minimum_split=2,
                 subsample=1.0, criterion='friedman_mse', minimum_samples_per_leaf=1,
                 weight_minimum_leaf_fraction=0.0, least_impurity_reduce=0.0, init=None, random_state=None,
                 how_many_features=None, verbose=0, greatest_node_of_leaf=None,
                 criteria_for_early_stopping=0.1, count_of_max_iteration=None, tol=0.0001, alpha_parameter=0.0):
        # It incorporates user-specified values or fallback defaults to define the model's operational characteristics.
        self.no_of_estimators = no_of_estimators
        self.best_iteration = None  # Tracks optimal iteration for early stopping
        self.criteria_for_early_stopping = criteria_for_early_stopping
        self.verbose = verbose
        self.subsample = subsample
        self.criterion = criterion
        self.initial_prediction = None  # Baseline prediction value
        self.alpha_parameter = alpha_parameter
        self.least_impurity_reduce = least_impurity_reduce
        self.minimum_samples_per_leaf = minimum_samples_per_leaf
        self.count_of_max_iteration = count_of_max_iteration
        self.init = init
        self.weight_minimum_leaf_fraction = weight_minimum_leaf_fraction
        self.models = []  # Stores the trained trees in the ensemble
        self.greatest_node_of_leaf = greatest_node_of_leaf
        self.how_many_features = how_many_features
        self.sample_minimum_split = sample_minimum_split
        self.random_state = random_state
        self.tol = tol
        self.rate_of_learning = rate_of_learning
        self.max_depth = max_depth

    # Configures random number generator with provided seed for reproducibility
    def _set_random_state(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        # Set up reproducibility for random operations
        self._set_random_state()

        # Initialize model predictions
        self.initial_prediction = self._initialize_predictions(y)

        predictions = np.full_like(y, self.initial_prediction)

        # Split data into training and validation sets if early stopping is enabled
        X_train, y_train, X_val, y_val = self._prepare_data_for_early_stopping(X, y)

        # Train the model
        for i in range(self.no_of_estimators):
            residuals = self._compute_residuals(y_train, predictions)
            X_sample, residuals_sample = self._subsample_data(X_train, residuals)

            tree = self._train_decision_tree(X_sample, residuals_sample)
            self.models.append(tree)

            predictions[:len(y_train)] = self._update_predictions(predictions, X_train, tree)

            if self.count_of_max_iteration:
                if self._early_stopping(X_val, y_val, predictions, i):
                    break

    def _initialize_predictions(self, y):
        if self.init is None:
            return np.mean(y)
        else:
            return self.init

    def _prepare_data_for_early_stopping(self, X, y):
        if self.count_of_max_iteration:
            n_val_samples = int(len(X) * self.criteria_for_early_stopping)
            X_train, X_val = X[:-n_val_samples], X[-n_val_samples:]
            y_train, y_val = y[:-n_val_samples], y[-n_val_samples:]
            return X_train, y_train, X_val, y_val
        else:
            return X, y, None, None

    def _compute_residuals(self, y_train, predictions):
        return y_train - predictions[:len(y_train)]

    def _train_decision_tree(self, X_sample, residuals_sample):
        tree = DecisionTree(
            max_depth=self.max_depth,
            sample_minimum_split=self.sample_minimum_split,
            criterion=self.criterion,
            minimum_samples_per_leaf=self.minimum_samples_per_leaf,
            weight_minimum_leaf_fraction=self.weight_minimum_leaf_fraction,
            least_impurity_reduce=self.least_impurity_reduce,
            how_many_features=self.how_many_features,
            greatest_node_of_leaf=self.greatest_node_of_leaf,
            alpha_parameter=self.alpha_parameter
        )
        tree.fit(X_sample, residuals_sample)
        return tree

    def _update_predictions(self, predictions, X_train, tree):
        return predictions + self.rate_of_learning * tree.predict(X_train)

    def _early_stopping(self, X_val, y_val, predictions, iteration):
        val_residuals = y_val - predictions[len(X_val):]
        val_loss = np.mean(np.square(val_residuals))
        if val_loss + self.tol < self.best_score:
            self.best_score = val_loss
            self.no_change_count = 0
            self.best_iteration = iteration
        else:
            self.no_change_count += 1
            if self.no_change_count >= self.count_of_max_iteration:
                if self.verbose:
                    print(f"Early stopping at iteration {iteration}")
                return True
        return False

    # Implements sub-sampling logic for data when subsample ratio is less than 1
    def _subsample_data(self, X, y):
        X = np.array(X)
        if self.subsample < 1.0:
            n_samples = int(self.subsample * X.shape[0])
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            return X[indices], y[indices]
        return X, y

    # Predicts target values for new data points using the trained ensemble
    def predict(self, X):
        X = np.array(X)
        predictions = np.full(X.shape[0], self.initial_prediction)  # Initialize predictions
        no_of_estimators = self.best_iteration + 1 if self.best_iteration else self.no_of_estimators
        for tree in self.models[:no_of_estimators]:  # Iterate only up to the best iteration if early stopping was used
            predictions += self.rate_of_learning * tree.predict(X)
        return predictions
