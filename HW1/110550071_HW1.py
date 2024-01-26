# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        # b = (xt x)-1 xt y
        X = np.c_[np.ones((len(X), 1)), X]
        beta =  np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.closed_form_intercept = beta[0]
        self.closed_form_weights = beta[1:]

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        
        cost_list=[]
        err_list=[]
        ones = np.ones((X.shape[0], 1))
        X = np.c_[ones, X]
        theta = np.random.randn(5, 1)
        # theta = np.zeros((5, 1))
        y = np.reshape(y, (len(y), 1))
        for i in range(epochs):
            y_pred = X.dot(theta)
            gradients = 2/len(X) * X.T.dot(y_pred-y)
            theta -= lr*gradients
            err = np.mean(np.square(y_pred-y))
            err_list.append(err)
            # theta = np.reshape(theta, (1, 5))
            # if ((i%10000)==0):     print(f"Epochs: {i}, Weights: {theta[0][1:]}, Intercept: {theta[0][0]}")
            # theta = np.reshape(theta, (5, 1))
        
        theta = np.reshape(theta, (1, 5))
        self.gradient_descent_intercept = theta[0][0]
        self.gradient_descent_weights = theta[0][1:]
        
        # plt.plot(range(2, epochs+1), err_list[1:], c="r", linewidth=2) # draw learning curve
        # plt.xlabel("epochs", fontsize=10)
        # plt.ylabel("MSE", fontsize=10)
        # plt.legend(["train"], loc="best")
        # plt.axis([2, epochs, 0, 100])
        # plt.show()

        

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        # Return the value.
        square_sum = np.square(np.subtract(prediction, ground_truth))
        return np.mean(square_sum)

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        # Return the prediction.
        y_pred = X.dot(self.closed_form_weights) + self.closed_form_intercept
        return y_pred

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        # Return the prediction.
        y_pred = X.dot(self.gradient_descent_weights) + self.gradient_descent_intercept
        return y_pred
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        
        pass

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.0001, epochs=1500000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")