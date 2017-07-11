#!/usr/bin/env python3

# IMPORTS
import tensorflow as tf
import numpy as np
##%matplotlib inline
from matplotlib import pyplot

# SETUP
tf.logging.set_verbosity(tf.logging.INFO)

# GLOBAL
# Training Data
x_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_data = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Hyperparameters
n_input = 2
n_hidden = 3
n_output = 1
learning_rate = 0.1
epochs = 10

# TF Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# TF Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# TF Bias
b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")


# TF Activation Functions
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y * tf.log(hy) - (1-Y) * tf.log(1 - hy))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)

