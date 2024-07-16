import numpy as np
import os
import cv2

# Load the training data from the folder
train_folder = '/home/vaishnavi-moutam/Downloads/final/Training1'
X_train = []
y_train = []
for file in os.listdir(train_folder):
    img = cv2.imread(os.path.join(train_folder, file))
    if img is not None:  # Check if image is not empty
        img = cv2.resize(img, (32, 32))  # Resize the image
        img = img.flatten()  # Flatten the image into a 1D array
        X_train.append(img)  # Append the flattened image to the list
        y_train.append(1)  # or 0, depending on the class label
    else:
        print(f"Error reading file {file}")  # Print an error message if the image is empty

# Convert the lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0

# Initialize the weights and bias for L2 regularization
weights_l2 = np.random.rand(X_train.shape[1])
bias_l2 = 0

# Initialize the weights and bias for L1 regularization
weights_l1 = np.random.rand(X_train.shape[1])
bias_l1 = 0

# Initialize the weights and bias for Dropout regularization
weights_dropout = np.random.rand(X_train.shape[1])
bias_dropout = 0

# Set the learning rate, regularization parameters, and number of iterations
learning_rate = 0.01
reg_param_l2 = 0.1  # L2 regularization parameter
reg_param_l1 = 0.01  # L1 regularization parameter
dropout_rate = 0.2  # Dropout rate
n_iters = 3

# Training loop with L2 regularization
for iter in range(n_iters):
    for i, x_i in enumerate(X_train):
        # Calculate the predicted output with L2 regularization
        predicted_l2 = np.where(np.dot(x_i, weights_l2) + bias_l2 >= 0, 1, -1)
        
        # Update the weights and bias with L2 regularization
        update_l2 = learning_rate * (y_train[i] - predicted_l2)
        weights_update_l2 = update_l2 * x_i - reg_param_l2 * weights_l2  # L2 regularization update
        weights_l2 += weights_update_l2
        bias_l2 += update_l2

    # Calculate the accuracy and loss with L2 regularization
    predictions_l2 = np.where(np.dot(X_train, weights_l2) + bias_l2 >= 0, 1, -1)
    accuracy_l2 = np.mean(predictions_l2 == y_train)
    loss_l2 = np.mean((predictions_l2 - y_train) ** 2) + 0.5 * reg_param_l2 * np.sum(weights_l2 ** 2)
    
    # Print the accuracy and loss with L2 regularization
    print(f'Iteration {iter+1}, L2 Regularization - Accuracy: {accuracy_l2:.2f}, Loss: {loss_l2:.2f}')

# Save the trained model with L2 regularization
np.save('weights_l2.npy', weights_l2)
np.save('bias_l2.npy', bias_l2)

# Training loop with L1 regularization
for iter in range(n_iters):
    for i, x_i in enumerate(X_train):
        # Calculate the predicted output with L1 regularization
        predicted_l1 = np.where(np.dot(x_i, weights_l1) + bias_l1 >= 0, 1, -1)
        
        # Update the weights and bias with L1 regularization
        update_l1 = learning_rate * (y_train[i] - predicted_l1)
        weights_update_l1 = update_l1 * x_i - reg_param_l1 * np.sign(weights_l1)  # L1 regularization update
        weights_l1 += weights_update_l1
        bias_l1 += update_l1

    # Calculate the accuracy and loss with L1 regularization
    predictions_l1 = np.where(np.dot(X_train, weights_l1) + bias_l1 >= 0, 1, -1)
    accuracy_l1 = np.mean(predictions_l1 == y_train)
    loss_l1 = np.mean((predictions_l1 - y_train) ** 2) + reg_param_l1 * np.sum(np.abs(weights_l1))
    
    # Print the accuracy and loss with L1 regularization
    print(f'Iteration {iter+1}, L1 Regularization - Accuracy: {accuracy_l1:.2f}, Loss: {loss_l1:.2f}')

# Save the trained model with L1 regularization
np.save('weights_l1.npy', weights_l1)
np.save('bias_l1.npy', bias_l1)

# Training loop with Dropout regularization
for iter in range(n_iters):
    for i, x_i in enumerate(X_train):
        # Apply dropout during training
        mask = np.random.binomial(1, 1 - dropout_rate, size=weights_dropout.shape)
        x_i_dropout = x_i * mask / (1 - dropout_rate)
        
        # Calculate the predicted output with Dropout regularization
        predicted_dropout = np.where(np.dot(x_i_dropout, weights_dropout) + bias_dropout >= 0, 1, -1)
        
        # Update the weights and bias without regularization (dropout already applied)
        update_dropout = learning_rate * (y_train[i] - predicted_dropout)
        weights_update_dropout = update_dropout * x_i_dropout
        weights_dropout += weights_update_dropout
        bias_dropout += update_dropout

    # Calculate the accuracy and loss with Dropout regularization
    predictions_dropout = np.where(np.dot(X_train, weights_dropout) + bias_dropout >= 0, 1, -1)
    accuracy_dropout = np.mean(predictions_dropout == y_train)
    loss_dropout = np.mean((predictions_dropout - y_train) ** 2)
    
    # Print the accuracy and loss with Dropout regularization
    print(f'Iteration {iter+1}, Dropout Regularization - Accuracy: {accuracy_dropout:.2f}, Loss: {loss_dropout:.2f}')

# Save the trained model with Dropout regularization
np.save('weights_dropout.npy', weights_dropout)
np.save('bias_dropout.npy', bias_dropout)

# Load the validation data from the folder
val_folder = '/home/vaishnavi-moutam/Downloads/final/Validation1'
X_val = []
y_val = []
for file in os.listdir(val_folder):
    img = cv2.imread(os.path.join(val_folder, file))
    if img is not None:  # Check if image is not empty
        img = cv2.resize(img, (32, 32))  # Resize the image
        img = img.flatten()  # Flatten the image into a 1D array
        X_val.append(img)  # Append the flattened image to the list
        y_val.append(1)  # or 0, depending on the class label
    else:
        print(f"Error reading file {file}")  # Print an error message if the image is empty

# Convert the lists to numpy arrays
X_val = np.array(X_val)
y_val = np.array(y_val)

# Normalize the pixel values to be between 0 and 1
X_val = X_val / 255.0

# Load the trained models with L1, L2, and Dropout regularization
weights_l2 = np.load('weights_l2.npy')
bias_l2 = np.load('bias_l2.npy')

weights_l1 = np.load('weights_l1.npy')
bias_l1 = np.load('bias_l1.npy')

weights_dropout = np.load('weights_dropout.npy')
bias_dropout = np.load('bias_dropout.npy')

# Evaluate the validation set accuracy for each regularization technique
predictions_l2 = np.where(np.dot(X_val, weights_l2) + bias_l2 >= 0, 1, -1)
accuracy_val_l2 = np.mean(predictions_l2 == y_val)
print(f'Validation Accuracy with L2 Regularization: {accuracy_val_l2:.2f}')

predictions_l1 = np.where(np.dot(X_val, weights_l1) + bias_l1 >= 0, 1, -1)
accuracy_val_l1 = np.mean(predictions_l1 == y_val)
print(f'Validation Accuracy with L1 Regularization: {accuracy_val_l1:.2f}')

predictions_dropout = np.where(np.dot(X_val, weights_dropout) + bias_dropout >= 0, 1, -1)
accuracy_val_dropout = np.mean(predictions_dropout == y_val)
print(f'Validation Accuracy with Dropout Regularization: {accuracy_val_dropout:.2f}')

# Load the testing data from the folder
test_folder = '/home/vaishnavi-moutam/Downloads/final/Testing1'
X_test = []
y_test = []
for file in os.listdir(test_folder):
    img = cv2.imread(os.path.join(test_folder, file))
    if img is not None:  # Check if image is not empty
        img = cv2.resize(img, (32, 32))  # Resize the image
        img = img.flatten()  # Flatten the image into a 1D array
        X_test.append(img)  # Append the flattened image to the list
        y_test.append(1)  # or 0, depending on the class label
    else:
        print(f"Error reading file {file}")  # Print an error message if the image is empty

# Convert the lists to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize the pixel values to be between 0 and 1
X_test = X_test / 255.0

# Evaluate the test set accuracy for each regularization technique
predictions_l2_test = np.where(np.dot(X_test, weights_l2) + bias_l2 >= 0, 1, -1)
accuracy_test_l2 = np.mean(predictions_l2_test == y_test)
print(f'Test Accuracy with L2 Regularization: {accuracy_test_l2:.2f}')

predictions_l1_test = np.where(np.dot(X_test, weights_l1) + bias_l1 >= 0, 1, -1)
accuracy_test_l1 = np.mean(predictions_l1_test == y_test)
print(f'Test Accuracy with L1 Regularization: {accuracy_test_l1:.2f}')

predictions_dropout_test = np.where(np.dot(X_test, weights_dropout) + bias_dropout >= 0, 1, -1)
accuracy_test_dropout = np.mean(predictions_dropout_test == y_test)
print(f'Test Accuracy with Dropout Regularization: {accuracy_test_dropout:.2f}')
