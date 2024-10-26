import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Update the path to point to my local "Downloads" directory
data_path = os.path.expanduser('~/Downloads/ML_hw3_attachment-cpusmall_scale.txt')

data = []
with open(data_path, 'r') as f:
    for line in f:
        values = line.split()
        label = float(values[0])
        features = np.array([float(val.split(':')[1]) for val in values[1:]])
        data.append((label, features))

# Convert to numpy arrays
data = np.array(data, dtype=object)
y = np.array([d[0] for d in data])
X = np.array([np.concatenate(([1], d[1])) for d in data])  # Add x_0 = 1 for the bias term

# SGD Parameters
eta = 0.01
num_iterations = 100000
record_interval = 200
num_experiments = 1126

# Storing the averaged error values
average_E_in = np.zeros(num_iterations // record_interval)
average_E_out = np.zeros(num_iterations // record_interval)
E_in_wlin_values = []
E_out_wlin_values = []
E_in_poly_values = []
E_out_poly_values = []

# Polynomial Transform (Q = 3)
def polynomial_transform(X, Q=3):
    X_poly = X.copy()
    for q in range(2, Q + 1):
        X_poly = np.hstack((X_poly, X[:, 1:] ** q))
    return X_poly

# Repeating the experiment with different random seeds
for experiment in range(num_experiments):
    # Randomly shuffle the data
    X_shuffled, y_shuffled = shuffle(X, y, random_state=experiment)
    
    # Take first 64 samples for training, rest for testing
    X_train, y_train = X_shuffled[:64], y_shuffled[:64]
    X_test, y_test = X_shuffled[64:], y_shuffled[64:]

    # Linear regression using pseudo-inverse to find w_lin
    w_lin = np.linalg.pinv(X_train).dot(y_train)
    
    # Calculate E_in(w_lin) and E_out(w_lin)
    E_in_wlin = np.mean((y_train - X_train.dot(w_lin)) ** 2)
    E_out_wlin = np.mean((y_test - X_test.dot(w_lin)) ** 2)
    E_in_wlin_values.append(E_in_wlin)
    E_out_wlin_values.append(E_out_wlin)
    
    # Polynomial transformation
    X_train_poly = polynomial_transform(X_train)
    X_test_poly = polynomial_transform(X_test)
    
    # Run linear regression to get w_poly
    w_poly = np.linalg.pinv(X_train_poly).dot(y_train)
    
    # Evaluate Ein(w_poly) and Eout(w_poly)
    E_in_poly = np.mean((y_train - X_train_poly.dot(w_poly)) ** 2)
    E_out_poly = np.mean((y_test - X_test_poly.dot(w_poly)) ** 2)
    E_in_poly_values.append(E_in_poly)
    E_out_poly_values.append(E_out_poly)

    # Initialize weights to zero for SGD
    w = np.zeros(X.shape[1])
    
    # Record error values at specific intervals
    E_in_values = []
    E_out_values = []
    
    # SGD loop
    for t in range(1, num_iterations + 1):
        # Pick one example randomly
        i = np.random.randint(0, len(X_train))
        x_i, y_i = X_train[i], y_train[i]
        
        # Compute gradient and update weights
        gradient = -2 * (y_i - np.dot(w, x_i)) * x_i
        w -= eta * gradient
        
        # Record errors every 'record_interval' iterations
        if t % record_interval == 0:
            E_in = np.mean((y_train - X_train.dot(w)) ** 2)
            E_out = np.mean((y_test - X_test.dot(w)) ** 2)
            E_in_values.append(E_in)
            E_out_values.append(E_out)
    
    # Sum the error values for averaging later
    average_E_in += np.array(E_in_values)
    average_E_out += np.array(E_out_values)

# Averaging the errors over all experiments
average_E_in /= num_experiments
average_E_out /= num_experiments
average_E_in_wlin = np.mean(E_in_wlin_values)
average_E_out_wlin = np.mean(E_out_wlin_values)
average_E_out_poly = np.mean(E_out_poly_values)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(range(record_interval, num_iterations + 1, record_interval), average_E_in, label='Average $E_{in}(w_t)$')
plt.plot(range(record_interval, num_iterations + 1, record_interval), average_E_out, label='Average $E_{out}(w_t)$')
plt.axhline(average_E_in_wlin, color='r', linestyle='--', label='Average $E_{in}(w_{lin})$')
plt.axhline(average_E_out_wlin, color='g', linestyle='--', label='Average $E_{out}(w_{lin})$')
plt.xlabel('Iteration (t)')
plt.ylabel('Error')
plt.legend()
plt.title('Average In-Sample and Out-of-Sample Errors for Linear Regression (SGD)')
plt.grid(True)
plt.show()

# Plot histogram of Ein(w_lin) - Ein(w_poly)
E_in_diff = np.array(E_in_wlin_values) - np.array(E_in_poly_values)
plt.figure(figsize=(12, 6))
plt.hist(E_in_diff, bins=30, edgecolor='k')
plt.xlabel('$E_{in}(w_{lin}) - E_{in}(w_{poly})$')
plt.ylabel('Frequency')
plt.title('Histogram of $E_{in}$ Gain for Polynomial Transform')
plt.show()

# Plot histogram of Eout(w_lin) - Eout(w_poly)
E_out_diff = np.array(E_out_wlin_values) - np.array(E_out_poly_values)
plt.figure(figsize=(12, 6))
plt.hist(E_out_diff, bins=30, edgecolor='k')
plt.xlabel('$E_{out}(w_{lin}) - E_{out}(w_{poly})$')
plt.ylabel('Frequency')
plt.title('Histogram of $E_{out}$ Gain for Polynomial Transform')
plt.show()

# Findings
description = """
Findings:
1. The histogram of $E_{in}(w_{lin}) - E_{in}(w_{poly})$ shows that, for most experiments, the polynomial transform leads to a reduction in in-sample error compared to the linear model.
2. The histogram of $E_{out}(w_{lin}) - E_{out}(w_{poly})$ shows the impact of the polynomial transformation on the generalization error. In many cases, the polynomial model also achieves a lower out-of-sample error, indicating better generalization.
3. This indicates that the polynomial model is able to better fit the training data due to the added complexity from higher-order terms. However, it is also more prone to overfitting, which can negatively impact its performance on unseen data.
"""
print(description)
