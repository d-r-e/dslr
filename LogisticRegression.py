import numpy as np
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html


class LogisticRegression:

    def __init__(self, weights=None, c=100, lr=0.01):
        self._weights = weights
        self._c = c
        self._lr = lr

    @staticmethod
    def sigmoid(X):
        return np.array(1 / (1 + np.exp(-X)))

    @staticmethod
    def predict(features, weights):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        z = np.dot(features, weights)
        return sigmoid(z)

    def fit(self, X, Y):
        return

    @staticmethod
    def cost_function(features, labels, weights):
        '''
        Using Mean Absolute Error

        Features:(100,3)
        Labels: (100,1)
        Weights:(3,1)
        Returns 1D matrix of predictions
        Cost =
            (x*log(yp) + (1-x)*log(1-yp) ) / len(x)
        '''
        observations = len(labels)
        predictions = predict(features, weights)
        # Take the error when label=1
        class1_cost = -labels*np.log(predictions)
        # Take the error when label=0
        class2_cost = (1-labels)*np.log(1-predictions)
        # Take the sum of both costs
        cost = class1_cost - class2_cost
        # Take the average cost
        cost = cost.sum() / observations
        return cost

    def update_weights(self, features, labels, weights, lr):
        '''
        Vectorized Gradient Descent

        Features:(200, 3)
        Labels: (200, 1)
        Weights:(3, 1)
        '''
        N = len(features)

        # 1 - Get Predictions
        predictions = predict(features, weights)
        # 2 Transpose features from (200, 3) to (3, 200)
        #  So we can multiply w the (200,1)  cost matrix.
        #  Returns a (3,1) matrix holding 3 partial derivatives --
        #  one for each feature -- representing the aggregate
        #  slope of the cost function across all observations
        gradient = np.dot(features.T,  predictions - labels)
        # 3 Take the average cost derivative for each feature
        gradient /= N
        # 4 - Multiply the gradient by our learning rate
        gradient *= self._lr
        # 5 - Subtract from our weights to minimize cost
        weights -= gradient
        return weights

    @staticmethod
    def decision_boundary(prob):
        return 1 if prob >= .5 else 0

    @staticmethod
    def classify(predictions):
        '''
        input  - N element array of predictions between 0 and 1
        output - N element array of 0s (False) and 1s (True)
        '''
        decision_boundary = np.vectorize(decision_boundary)
        return decision_boundary(predictions).flatten()

    def train(self, features, labels, weights):
        lr = self._lr
        iters = self._c
        cost_history = []

        for i in range(iters):
            weights = update_weights(features, labels, weights, lr)

            #Calculate error for auditing purposes
            cost = cost_function(features, labels, weights)
            cost_history.append(cost)

            # Log Progress
            if i % 1000 == 0:
                print("iter: "+str(i) + " cost: "+ str(cost))
        self._weights = weights
        return weights, cost_history

if __name__ == "__main__":
    X = np.array(np.arange(-5, 5))
    Y = X.copy()
    lr = LogisticRegression()
    print(lr.predict(X))
