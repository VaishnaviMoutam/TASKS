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

# Initialize the weights and bias
weights = np.random.rand(X_train.shape[1])
bias = 0

# Set the learning rate and number of iterations
learning_rate = 0.01
n_iters = 1

# Train the model
for iter in range(n_iters):
    for i, x_i in enumerate(X_train):
        # Calculate the predicted output
        predicted = np.where(np.dot(x_i, weights) + bias >= 0, 1, -1)
        # Update the weights and bias
        update = learning_rate * (y_train[i] - predicted)
        weights += update * x_i
        bias += update
    # Calculate the accuracy and loss
    accuracy = np.mean(predicted == y_train)
    loss = np.mean((predicted - y_train) ** 2)
    # Print the accuracy and loss at each iteration
    print(f'Iteration {iter+1}, Accuracy: {accuracy:.2f}, Loss: {loss:.2f}')

# Save the trained model
np.save('weights.npy', weights)
np.save('bias.npy', bias)

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

# Load the trained model
weights = np.load('weights.npy')
bias = np.load('bias.npy')

# Make predictions on the validation set
predictions = np.where(np.dot(X_val, weights) + bias >= 0, 1, -1)
# Calculate the accuracy on the validation set
accuracy = np.mean(predictions == y_val)
print(f'Validation Accuracy: {accuracy:.2f}')

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

# Load the trained model
weights = np.load('weights.npy')
bias = np.load('bias.npy')

# Make predictions on the testing set
predictions = np.where(np.dot(X_test, weights) + bias >= 0, 1, -1)
# Calculate the accuracy on the testing set
accuracy = np.mean(predictions == y_test)
print(f'Testing Accuracy: {accuracy:.2f}')

