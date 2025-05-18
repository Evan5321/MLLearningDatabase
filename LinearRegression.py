import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        np.random.seed(45)

        self.x = np.linspace(0, 10, 100)
        self.true_slope = 2 # True slope of the line
        self.true_intercept = 1 # True intercept of the line
        self.noise = np.random.normal(0, 2, 100) # Noise to add to the data
        self.y = self.true_slope * self.x + self.true_intercept + self.noise # True y values with noise


    def model(self, x, slope, intercept): # Define the model function
        return slope * x + intercept # Return the predicted y values
    
    def loss(self, y_true, y_pred): # Define the loss function
        return np.mean((y_true - y_pred) ** 2) # Return the mean squared error
    
    def gradient(self, x, y_true, y_pred): # Define the gradient function
        d_slope = -2 * np.mean(x * (y_true - y_pred)) # Calculate the derivative of the slope
        d_intercept = -2 * np.mean(y_true - y_pred) # Calculate the derivative of the intercept
        return d_slope, d_intercept # Return the derivatives

    def train(self, x, y_true, epochs=1000, learning_rate=0.01): # Define the train function
        slope = np.random.randn() # Initialize the slope
        intercept = np.random.randn() # Initialize the intercept

        for epoch in range(epochs): # Loop through the epochs
            y_pred = self.model(x, slope, intercept) # Calculate the predicted y values
            loss = self.loss(y_true, y_pred) # Calculate the loss
            d_slope, d_intercept = self.gradient(x, y_true, y_pred) # Calculate the derivatives
            slope -= learning_rate * d_slope # Update the slope
            intercept -= learning_rate * d_intercept # Update the intercept

            if epoch % 100 == 0: # Print the loss every 100 epochs
                print(f'Epoch {epoch}, Loss: {loss}') # Print the epoch and loss
        return slope, intercept # Return the slope and intercept
    
    def predict(self, x, slope, intercept): # Define the predict function
        return self.model(x, slope, intercept) # Return the predicted y values
    
    def plot(self, x, y_true, slope, intercept): # Define the plot function
        #plt.scatter(x, y_true, label='Data with Noise') # Plot the data with noise
        plt.plot(x, slope * x + intercept, color='green', label='Predicted Line') # Plot the predicted lineis
        plt.scatter(self.x, self.y, label='Data with Noise') # Plot the data with noise
        plt.plot(self.x, self.true_slope * self.x + self.true_intercept, color='red', label='True Line') # Plot the true line
        plt.xlabel('X') # Label the x-axis
        plt.ylabel('Y') # Label the y-axis
        plt.title('Linear Regression Example') # Title the plot
        plt.legend() # Show the legend
        plt.show() # Show the plot

if __name__ == '__main__':
    lr = LinearRegression() # Create an instance of the LinearRegression class
    print('Done') # Print a message to indicate that the program has finished running
    x_new = 3.5
    y_pred = lr.predict(x_new, lr.train(lr.x, lr.y)[0], lr.train(lr.x, lr.y)[1]) # Predict the y value for a new x value
    print(f'Predicted y value for x = {x_new}: {y_pred}') # Print the predicted y value for the new x value
    lr.plot(lr.x, lr.y, lr.train(lr.x, lr.y)[0], lr.train(lr.x, lr.y)[1]) # Plot the data and the predicted line