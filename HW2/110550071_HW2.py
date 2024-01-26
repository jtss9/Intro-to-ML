# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None
    
    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.intercept = 0

        for i in range(self.iteration):
            scores = np.dot(X, self.weights) + self.intercept
            pred = self.sigmoid(scores)
            error = y - pred
            gradient = np.dot(X.T, error)
            self.weights += self.learning_rate * gradient / y.size
            self.intercept += self.learning_rate * np.sum(error) / y.size
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        probabilities = self.sigmoid(np.dot(X, self.weights) + self.intercept)
        predictions = np.where(probabilities > 0.5, 1, 0)
        return predictions

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        X0 = X[y==0]
        X1 = X[y==1]
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)
        self.sw = np.dot((X0-self.m0).T, (X0-self.m0)) + np.dot((X1-self.m1).T, (X1-self.m1))
        self.sb = np.outer((self.m0-self.m1), (self.m0-self.m1))
        eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(self.sw), self.sb))
        self.w = eigenvectors[:, np.argmax(eigenvalues)]
        
        # self.sb = np.dot((self.m0-self.m1).reshape(-1,1), (self.m0-self.m1).reshape(1,-1))
        # self.w = np.dot(np.linalg.inv(self.sw), (self.m0-self.m1).reshape(-1,1))
        self.slope = self.w[1]/self.w[0]
    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        projected = np.dot(X, self.w)
        projected_m0 = np.dot(self.m0, self.w)
        projected_m1 = np.dot(self.m1, self.w)
        y_pred = np.where(np.abs(projected-projected_m0) < np.abs(projected-projected_m1), 0, 1)
        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        intercept = (self.m0[1]+self.m1[1])/2 - self.slope*(self.m0[0]+self.m1[0])/2
        for x in X:
            # x: (x0, x1)
            # eq1: y = slope * x + intercept
            # eq2: y = -1/slope *x + b
            b = x[1] + 1/self.slope * x[0]
            x_intersect = (b - intercept) / (self.slope + 1/self.slope)
            y_intersect = self.slope * x_intersect + intercept
            plt.plot((x[0], x_intersect), (x[1], y_intersect), color="blue", linestyle="--", alpha = 0.5)
            pass
        plt.axline((0, intercept), slope=self.slope, color="blue")
        plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=plt.cm.Paired)
        plt.xlabel("age")
        plt.ylabel("thalach")
        plt.title(f"Projection Line: slope={self.slope}, intercept={intercept}")
        plt.axis("equal")
        plt.xlim(0, 100)
        plt.ylim(80, 200)
        plt.savefig("FLD.png")
        plt.show()
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.0005, iteration=150000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    # FLD.plot_projection(X_test)

