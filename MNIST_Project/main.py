import numpy as np
from sklearn.datasets import fetch_openml

# mnist = fetch_openml('mnist_784',parser='auto')

# # Save the data and target arrays to a compressed NumPy binary file
# np.savez_compressed('mnist.npz', data=mnist['data'], target=mnist['target'])

# Load the MNIST data from the compressed NumPy binary file
with np.load('mnist.npz', allow_pickle=True) as data:
    X = data['data']  # pixel values for each image
    y = data['target']  # target class labels for each image

# Print the shape of the data and target arrays
print('Data shape:', X.shape)
print('Target shape:', y.shape)

# print(mnist.keys())