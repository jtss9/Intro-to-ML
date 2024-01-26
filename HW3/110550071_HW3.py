# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y)) / len(y)
    class_weights = np.bincount(y, weights=sample_weight)
    total_weight = np.sum(sample_weight)
    gini = 1.0 - np.sum((class_weights / total_weight) ** 2)
    return gini
    

# This function computes the entropy of a label array.
def entropy(y, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y)) / len(y)
    class_weights = np.bincount(y, weights=sample_weight)
    total_weight = np.sum(sample_weight)
    non_zero_elements = class_weights[class_weights != 0]
    probabilities = non_zero_elements / total_weight
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold for the split
        self.left = left
        self.right = right
        self.value = value  # Prediction value for leaf
    
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.tree = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y, sample_weight):
        if self.criterion == 'gini':
            return gini(y, sample_weight)
        elif self.criterion == 'entropy':
            return entropy(y, sample_weight)
    
    def build_tree(self, X, y, depth, sample_weight):
        if depth == self.max_depth or len(set(y)) == 1:
            return Node(value=max(set(y), key=list(y).count))
        
        num_features = X.shape[1]
        best_feature = 0
        best_gini = float('inf')
        best_threshold = 0
        
        for f_index in range(num_features):
            thresholds = set(X[:, f_index])
            for t in thresholds:
                left_i = X[:, f_index] <= t
                right_i = ~left_i
                
                left_gini = self.impurity(y[left_i], sample_weight[left_i])
                right_gini = self.impurity(y[right_i], sample_weight[right_i])
                weighted_gini = (left_gini*len(y[left_i]) + right_gini*len(y[right_i]))/len(y)
                
                if weighted_gini < best_gini:
                    best_feature = f_index
                    best_gini = weighted_gini
                    best_threshold = t
                    
        if best_gini == float('inf'):
            return Node(value=max(set(y), key=list(y).count))
        
        left_child = self.build_tree(X[X[:, best_feature] <= best_threshold], y[X[:, best_feature] <= best_threshold], depth+1, sample_weight[X[:, best_feature] <= best_threshold])
        right_child = self.build_tree(X[X[:, best_feature] > best_threshold], y[X[:, best_feature] > best_threshold], depth+1, sample_weight[X[:, best_feature] > best_threshold])
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        self.tree = self.build_tree(X, y, depth=0, sample_weight=sample_weight)
        
    def _predict(self, node, inputs):
        if node.value != None:
            return node.value
        if inputs[node.feature] <= node.threshold:
            return self._predict(node.left, inputs)
        return self._predict(node.right, inputs)
        
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        return [self._predict(self.tree, inputs) for inputs in X]
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        feature_importance = np.zeros(len(columns))

        def get_importance(node, importance):
            if node.feature != None:
                importance[node.feature] += 1
                get_importance(node.left, importance)
                get_importance(node.right, importance)

        get_importance(self.tree, feature_importance)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(columns, feature_importance)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []
        self.clfs = []

    def resample(self, X, y, sample_weight):
        sample_index = np.random.choice(len(X), len(X), replace=True, p=sample_weight)
        sample_index.sort()
        X_new = []
        y_new = []
        for i in sample_index:
            X_new.append(X[i])
            y_new.append(y[i])
        return np.array(X_new), np.array(y_new)
    
    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            X, y = self.resample(X, y, w)
            tree = DecisionTree(criterion=self.criterion, max_depth=1)
            tree.fit(X, y, sample_weight=w)
            y_pred = tree.predict(X)
            err = np.sum(w * (y_pred != y))
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)
            
            self.alphas.append(alpha)
            self.clfs.append(tree)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = np.zeros(len(X))
        for alpha, clf in zip(self.alphas, self.clfs):
            predictions += alpha * np.array(clf.predict(X))
        return np.sign(predictions).astype(int)

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(121)
# 26 3 0.78
# 81 1 0.85
#121 2 0.80
# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    # tree.plot_feature_importance_img(train_df.columns[:-1])

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=2)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
