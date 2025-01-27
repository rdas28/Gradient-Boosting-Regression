# Gradient Boosting Regression

## Table of Contents
1. Project Overview
2. Key Features
3. Changing Parameters via Command Line
4. Code Explanation
5. Adjustable Parameters
6. Known Limitations
7. Contributors

---

## Project Overview
Gradient Boosting is a powerful machine learning method used for both regression and classification tasks. It builds a strong predictive model by combining multiple weak learners, where each subsequent model is trained to reduce the loss function—such as mean squared error or cross-entropy—of the previous model using gradient descent. This project demonstrates a Gradient Boosting Regressor built entirely from scratch using only Python and NumPy. It is designed to provide a hands-on understanding of Gradient Boosting without relying on external machine learning libraries like scikit-learn, making it both a practical implementation and a valuable learning resource.

---

## Key Features

### Custom Implementation
- Custom implementation of base learners (Decision Trees) designed specifically for regression tasks, developed entirely from scratch without relying on external libraries for tree-based learning.
- Gradient Boosting framework constructed incrementally to solve real-world regression problems, where each decision tree is trained sequentially to minimize residual errors, and predictions are iteratively refined at each step

### Advanced Techniques

- Includes support for early stopping by monitoring validation loss to prevent overfitting.
- Implements Stochastic Gradient Boosting through subsampling of the training data.

### Parameter Customization
#### Fully customizable hyperparameters:

1. **Model Parameters**:
- ***number of estimators***: Increasing this value enhances model capacity but may result in overfitting if early stopping is not applied.
- ***learning rate***: Smaller learning rates improve generalization but require a higher number of estimators.
- ***max depth***: Limiting the depth of the trees helps to control model complexity.
- ***criterion***: Specifies the splitting criterion for the trees (default: friedman_mse).

2. **Regularization Parameters**:
- ***subsampling***: Using a fraction of the training data for each estimator adds randomness, improving generalization but potentially increasing bias.
- ***sample minimum split***: Sets the minimum number of samples required to split an internal node.
- ***minimum samples per leaf***: Determines the minimum number of samples needed at a leaf node. Larger leaf sizes help reduce overfitting.
- ***alpha parameter***: Serves as a regularization term to control tree complexity, reducing the risk of overfitting but possibly missing finer data patterns.
- **least impurity reduce**: Prevents unnecessary splits by ensuring only meaningful reductions in impurity lead to splits.

3. **Stopping Criteria**:
- ***count of max iteration***: Stops training when validation loss shows no improvement.
- ***early stopping***: Enhances efficiency and promotes generalization

---

## Changing Parameters via Command Line

The script allows you to modify the model's hyperparameters and test data file directly from the command line. Here's how to specify each parameter:

### Examples

1. **Use Default Parameters:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel
   ```

2. **Specify Custom Learning Rate and File Path:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel --file_path "gradientboosting/tests/test_data.csv" --rate_of_learning 0.05
   ```

3. **Change Multiple Parameters:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel --no_of_estimators 200 --max_depth 3 --sample_minimum_split 5 --criteria_for_early_stopping 0.1
   ```
---

### Adjustable Parameters:

| Parameter                  | Default Value                 | Description                                                                                   | Possible Values/Explanation                                                      |
|----------------------------|-------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `--file_path`              | `"gradientboosting/tests/test_data.csv"` | The path to the CSV file that contains the test data.                                         | Provide the path to any CSV file.                                               |
| `--no_of_estimators`       | `1000`                        | Defines how many boosting rounds will be performed to build the model.                       | Any integer value (e.g., 100, 500, 2000).                                       |
| `--rate_of_learning`       | `0.2`                         | Controls how much each tree contributes to the final prediction.                              | Any float value (e.g., 0.05, 0.1, 0.3).                                         |
| `--max_depth`              | `2`                           | Limits how deep each decision tree can grow.                                                  | Any integer value (e.g., 3, 4, 5, 6).                                           |
| `--sample_minimum_split`   | `2`                           | The smallest number of samples needed to split a node.                                        | Any integer value (e.g., 5, 10, 15).                                            |
| `--subsample`              | `1.0`                         | The percentage of the dataset used to train each tree.                                        | Any float between 0 and 1 (e.g., 0.8, 0.9).                                     |
| `--criterion`              | `"friedman_mse"`              | The metric used to decide how to split nodes during training.                                 | `friedman_mse`, `mse`, `mae` (Mean Squared Error or Mean Absolute Error).        |
| `--minimum_samples_per_leaf`| `1`                          | The minimum number of samples that must be in a leaf node.                                    | Any integer value (e.g., 2, 4).                                                 |
| `--weight_minimum_leaf_fraction` | `0.0`                   | The smallest fraction of sample weights allowed in a leaf node.                              | Any float value between 0 and 1 (e.g., 0.1, 0.2).                                |
| `--least_impurity_reduce`  | `0.0`                         | Specifies the minimum impurity reduction needed to split a node.                              | Any float value (e.g., 0.01, 0.05).                                             |
| `--random_state`           | `None`                        | A seed value to make the results reproducible.                                                | Any integer value (e.g., 42, 12345) or `None`.                                  |
| `--how_many_features`      | `None`                        | Determines how many features should be considered when looking for the best split.           | `None`, `sqrt`, `log2`, or an integer value (e.g., 5).                          |
| `--verbose`                | `0`                           | Adjusts the amount of information displayed during the training process.                      | Any integer value (e.g., 1 for less output, 10 for more detailed debugging).     |
| `--greatest_node_of_leaf`  | `None`                        | Sets the maximum number of leaf nodes allowed in a single tree.                               | Any integer value (e.g., 10, 50, 100).                                          |
| `--criteria_for_early_stopping` | `0.1`                    | Defines how much of the training data is used for validation in early stopping.               | Any float between 0 and 1 (e.g., 0.2).                                          |
| `--count_of_max_iteration` | `None`                        | Specifies how many iterations can pass without improvement before stopping early.             | Any integer value (e.g., 10, 20).                                               |
| `--tol`                    | `0.0001`                      | A small number that decides when to stop training due to lack of progress.                    | Any float value (e.g., 0.001, 0.0005).                                          |
| `--alpha_parameter`        | `0.0`                         | A parameter that helps prevent overfitting by pruning trees based on complexity.              | Any float value (e.g., 0.01, 0.1).                                              |

---


## Code Explanation 
1. **GradientBoostingRegressor**:
   ## Implements Gradient Boosting logic:
   1. **Hyperparameter Setup**: The model starts by configuring the hyperparameters and initializing the necessary attributes for the Gradient Boosting framework.
   2. **Seed Initialization**: A random seed is set using np.random.seed to guarantee reproducibility of the results.
   3. **Subsampling Data**: If subsampling is enabled, a random subset of rows is selected from the dataset; otherwise, the entire dataset is utilized for training.
   4. **Model Training (fit)**: The model begins by initializing the predictions, either using the provided init value or the mean of the target variable. If early stopping is enabled, the dataset is split into training and validation subsets. For each estimator, residuals are calculated as the difference between the actual and predicted values. When subsampling is enabled, a random subset of the data is selected for training. A decision tree is then trained on the residuals, added to the list of estimators, and the predictions are updated by incorporating the tree's output scaled by the learning rate. This iterative process repeats for the predefined number of trees.
   5. **Prediction**: After training, the model generates predictions for new data points by utilizing the trained ensemble of decision trees.

 
2. **DecisionTree**:
   - ## Implements CART (Classification and Regression Trees) from scratch.
   1. **Initialization**: The model initializes various hyperparameters, such as max_depth, sample_minimum_split, criterion, and others. If a random state is specified, the seed is set using np.random.seed to ensure consistent and reproducible results.

   2. **Mean Squared Error (MSE)**: Computes the variance of the target values, which acts as the impurity measure used for determining decision tree splits.

   3. **Impurity Calculation**: It employs Mean Squared Error (MSE) as the impurity metric to guide the tree's decision-making process.

   4. **Data Splitting**: The data is split into two subsets based on a threshold value of a chosen feature. The method returns the feature values and target values for each subset 

   5. **Best Split Selection**: The _best_split method evaluates all features and potential threshold values to identify the optimal split that minimizes impurity, measured by Mean Squared Error (MSE). For each split, it randomly selects a subset of features, and for each chosen feature, it examines all possible thresholds to determine the one that achieves the greatest impurity reduction.

   6. **Tree Building**: The decision tree is constructed recursively by identifying the optimal split and dividing the data into left and right subsets. This process continues until one of the following conditions is met: the maximum tree depth is reached, the node contains a single class, or the number of samples is too small for further splitting. If no valid split is identified, a leaf node is created, assigning it the mean of the target values.

   7. **Pruning**: This method streamlines the tree by merging nodes using a cost-complexity criterion. It recursively prunes the left and right subtrees, and if merging results in a reduced total cost, the nodes are merged. Pruning is applied when the total cost becomes less than the specified complexity parameter, alpha_parameter.

   8. **Training (fit):** The fit method trains the decision tree by recursively constructing it with the _build_tree method. If the alpha_parameter is set to a value greater than 0, the tree is pruned using the _prune method to prevent overfitting.

   9. **Prediction**: The _predict method navigates through the trained decision tree to determine the target value for a single data point. It follows the tree's branches based on the data point's feature values and returns the predicted value upon reaching a leaf node.

   10. **Predicting for Multiple Data Points**: The predict method produces predictions for all input data points by invoking the _predict method for each data point in the input matrix X. It returns an array containing the predicted values.

### Key Methods
- `fit(X, y)`: Trains the Gradient Boosting model by iteratively adding decision trees to reduce residuals.

- `predict(X)`: Makes predictions by combining the outputs of all decision trees.


## Model Evaluation Metrics for Gradient Boosting Model

1. **Coefficient of Determination (R² Score)**
-The R² Score evaluates the proportion of variance in the target variable that can be explained by the independent variables. It provides a measure of how closely the model's predictions match the actual data, indicating the model's overall goodness of fit.

- A high R² Score signifies that the model effectively captures the underlying patterns within the data.
Significance: This metric reflects the overall fit of the model and indicates the proportion of variance in the target variable that is explained by the input features.
2. **Mean Squared Error (MSE)**
MSE (Mean Squared Error) computes the average of the squared differences between the actual and predicted values.

- Interpretation: This metric is highly sensitive to large prediction errors, as squaring the differences emphasizes them.
- Significance: MSE highlights significant deviations in predictions, making it useful for identifying whether the model is overfitting or underfitting. It provides insight into how well the model captures the data without being overly influenced by extreme errors.
3. **Mean Absolute Error (MAE)**
MAE (Mean Absolute Error) calculates the average magnitude of errors between the actual and predicted values without squaring them, giving equal weight to all errors.

- Interpretation: Unlike MSE, MAE treats all deviations equally, making it a useful metric for understanding the typical size of prediction errors.
- Significance: A low MAE indicates that, on average, the model's predictions are close to the actual values. Since MAE is less sensitive to large errors, it complements MSE by providing a more balanced perspective on the model's performance.
4. **Root Mean Squared Error (RMSE)**
RMSE (Root Mean Squared Error) is the square root of MSE, representing the average prediction error in the same units as the target variable.
- Interpretation: RMSE provides a clear and interpretable measure of the model's prediction errors, directly comparable to the scale of the target variable.
- Significance: A low RMSE suggests that the model's predictions are closely aligned with the actual values, reflecting both reliability and accuracy.

---
## Sample Results:

| Metric                  | Value    |
|-------------------------|----------|
| **R² Score**            | 0.9876   |
| **Mean Squared Error**  | 0.0003   |
| **Mean Absolute Error** | 0.0123   |
| **Root Mean Squared Error** | 0.0175 |


--- 


## Known Limitations
### 1. **High-Dimensional Data**
- **Problem**: Higher computational demands and a greater likelihood of overfitting.
- **Solutions**:
  - Feature selection.
  - Dimensionality reduction (e.g., PCA).
  - Reducing the number of features considered per split (`how_many_features`).

### 2. **Imbalanced Datasets**
- **Problem**: Sensitivity to imbalanced target values.
- **Solutions**:
  - Adjusting sample weights.
  - Using specialized loss functions that are robust to outliers.

### 3. **Extremely Large Datasets**
- **Problem**: Excessive training time.
- **Solutions**:
  - Implementing parallel processing for tree building.
  - Increasing `subsample` to use smaller subsets of data.

### 4. **Noisy or Inconsistent Data**
- **Problem**: Overfitting to noise in the data.
- **Solutions**:
  - Aggressive pruning.
  - Adjusting `alpha_parameter` for complexity control.
  - Limiting tree depth (`max_depth`).
  - Increasing the `min_samples_leaf` parameter.

### 5. **Categorical Variables**
- **Problem**: Inefficient handling of categorical data.
- **Solutions**:
  - Automatic encoding of categorical features.
  - Implementing custom splitting criteria for categorical variables.

### 6. **Extrapolation**
- **Problem**: Poor performance on inputs outside the training data range.
- **Solutions**:
  - Ensuring training data covers the expected input range.
  - Combining with models better suited for extrapolation.

Addressing these challenges enhances the model's robustness and widens its applicability across diverse datasets.

---

## Contributors
- Riddhi Das (A20582829): rdas8@hawk.iit.edu
- Madhur Gusain (A20572395): mgusain@hawk.iit.edu




