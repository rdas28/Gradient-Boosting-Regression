import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # Container for split criteria, branches, and predicted value for a node
        self.feature_index = feature_index  # Index of the feature chosen for splitting
        self.threshold = threshold  # Value determining the split point
        self.left = left  # Node's left child branch
        self.right = right  # Node's right child branch
        self.value = value  # Output for leaf nodes (e.g., regression target)

class DecisionTree:
    def __init__(self, max_depth=5, sample_minimum_split=2, minimum_samples_per_leaf=1, 
                 weight_minimum_leaf_fraction=0.0, least_impurity_reduce=0.0, how_many_features=None, 
                 random_state=None, greatest_node_of_leaf=None, alpha_parameter=0.0, criterion='friedman_mse'):
        # Core parameters for controlling tree growth and optimization
        self.max_depth = max_depth  # Max allowable depth of the tree
        self.sample_minimum_split = sample_minimum_split  # Min data points to split a node
        self.minimum_samples_per_leaf = minimum_samples_per_leaf  # Min samples in a leaf node
        self.weight_minimum_leaf_fraction = weight_minimum_leaf_fraction  # Fractional weight threshold for leaves
        self.least_impurity_reduce = least_impurity_reduce  # Min impurity reduction to justify a split
        self.how_many_features = how_many_features  # Number of features considered per split
        self.random_state = random_state  # Seed for reproducibility
        self.greatest_node_of_leaf = greatest_node_of_leaf  # Optional max leaf-specific node limit
        self.alpha_parameter = alpha_parameter  # Post-pruning complexity regularization factor
        self.criterion = criterion  # Splitting metric, e.g., 'friedman_mse'
        self.root = None  # Root node of the decision tree

        # Ensuring deterministic behavior if a random state is defined
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _mean_squared_error(self, y):
        """Calculate variance as a proxy for impurity."""
        return np.mean((y - np.mean(y)) ** 2)
    def _best_split(self, X, y):
        """Locate the optimal feature and threshold to minimize impurity."""
        X = np.array(X)
        best_feature, best_threshold, best_mse = None, None, float('inf')
        parent_mse = self._mean_squared_error(y)  # Baseline impurity
        n_features = X.shape[1]  # Total feature count

        # Define number of features to consider at each split
        if isinstance(self.how_many_features, str):
            if self.how_many_features == 'sqrt':
                how_many_features = int(np.sqrt(n_features))
            elif self.how_many_features == 'log2':
                how_many_features = int(np.log2(n_features))
            else:
                raise ValueError(f"Invalid value for how_many_features: {self.how_many_features}")
        elif self.how_many_features is None:
            how_many_features = n_features
        else:
            how_many_features = self.how_many_features

        # Randomly sample features for potential splits
        features = np.random.choice(n_features, how_many_features, replace=False)

        # Test each threshold for each selected feature
        for feature_index in features:
            thresholds = np.unique(X[:, feature_index])  # Possible splits
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)

                # Skip invalid splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Compute child node impurity
                left_mse = self._mean_squared_error(y_left)
                right_mse = self._mean_squared_error(y_right)
                child_mse = (len(y_left) / len(y)) * left_mse + (len(y_right) / len(y)) * right_mse

                # Update optimal split if improvement found
                if child_mse < best_mse:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_mse = child_mse

        return best_feature, best_threshold
    
    def _compute_criterion(self, y):
        """Determine impurity metric based on the criterion selection."""
        if self.criterion == 'friedman_mse':
            return self._mean_squared_error(y)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        
    def _prune(self, node, alpha):
        """On cost-complexity tree is simplified by merging nodes."""
        if node.left is None and node.right is None:  # Leaf node
            return node.value, 1, alpha * 1

        # Prune left and right subtrees recursively
        left_value, left_leaves, left_cost = self._prune(node.left, alpha) if node.left else (0, 0, 0)
        right_value, right_leaves, right_cost = self._prune(node.right, alpha) if node.right else (0, 0, 0)

        # Evaluate merge condition
        if left_cost + right_cost <= alpha * (1 + left_leaves + right_leaves):
            return (left_value + right_value) / (left_leaves + right_leaves), 1, alpha * 1

        return node, left_leaves + right_leaves, left_cost + right_cost
    def _split(self, X, y, feature_index, threshold):
        """Partition data into two subsets based on threshold."""
        X = np.array(X)
        left_idx = np.where(X[:, feature_index] <= threshold)  # IDs for left subset
        right_idx = np.where(X[:, feature_index] > threshold)  # IDs for right subset
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    def _build_tree(self, X, y, depth):
        """Recursively create the decision tree structure."""
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or 
            len(y) < self.sample_minimum_split):  # Stop growing tree
            return DecisionTreeNode(value=np.mean(y))  # Leaf node value

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:  # No valid split found
            return DecisionTreeNode(value=np.mean(y))

        # Recursive building of subtrees
        X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        return DecisionTreeNode(feature_index, threshold, left_subtree, right_subtree)

    def _predict(self, x, node):
        """Traverse the tree to predict a single instance."""
        if isinstance(node, DecisionTreeNode):
            if node.value is not None:  # Leaf node reached
                return node.value
            if x[node.feature_index] <= node.threshold:  # Follow left branch
                return self._predict(x, node.left)
            else:  # Follow right branch
                return self._predict(x, node.right)
        return node  # Handle pruned leaf
    
    def fit(self, X, y):
        """Train the decision tree using the input data."""
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, 0)  # Construct tree
        if self.alpha_parameter > 0.0:  # Apply pruning if required
            self.root, _, _ = self._prune(self.root, self.alpha_parameter)


    def predict(self, X):
        """Generate predictions for all data points."""
        return np.array([self._predict(x, self.root) for x in X])
