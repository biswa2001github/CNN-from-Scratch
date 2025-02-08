import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from dense import Dense
from conv import Convolutional
from reshape import Reshape
from activation_functions import sigmoid  # Ensure this is a class-based layer
from loss_functions import binary_cross_entropy, binary_cross_entropy_derivative


def data_processing(x, y, limit):
    """Processes MNIST data to filter specific classes and reshape for CNN."""
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]

    all_indices = np.hstack((zero_index, one_index))
    np.random.shuffle(all_indices)

    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28).astype('float32') / 255  # Normalize
    
    # One-hot encode and select only columns for digits 0 and 1
    y = to_categorical(y, num_classes=10)[:, [0, 1]]
    y = y.reshape(len(y), 2, 1)  # Shape for network output

    return x, y


# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = data_processing(x_train, y_train, 200)
x_test, y_test = data_processing(x_test, y_test, 80)

# Define CNN model architecture
network = [
    Convolutional((1, 28, 28), kernel_size=3, depth=5),
    sigmoid(),  # Use a class-based activation function
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    sigmoid(),
    Dense(100, 2),
    sigmoid()
]

# Training parameters
epochs = 10
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    error = 0

    for x, y in zip(x_train, y_train):
        # Forward pass
        output = x
        for layer in network:
            output = layer.forward(output)

        # Compute loss
        error += binary_cross_entropy(y, output)

        # Backward pass
        gradient = binary_cross_entropy_derivative(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

    # Print loss per epoch
    error /= len(x_train)
    print(f"Epoch: {epoch + 1}, Loss: {error:.3f}")

# Testing phase
accuracy = 0
print("\nTest set predictions:")
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    # Accuracy in test set
    if np.argmax(output) == np.argmax(y):
        accuracy += 1
    print(f"Prediction: {np.argmax(output)}, True: {np.argmax(y)}")

print(f"\nAccuracy in test set: {accuracy/len(y_test):.2f}")
