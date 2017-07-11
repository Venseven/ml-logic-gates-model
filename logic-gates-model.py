
# coding: utf-8

# #  Imports

# In[1]:


# IMPORTS
import tensorflow as tf
import numpy as np
##get_ipython().magic('matplotlib inline')
##from matplotlib import pyplot

# SETUP
tf.logging.set_verbosity(tf.logging.INFO)

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


# # Global

# In[2]:


# GLOBAL

# Constants
L_XOR = 0
L_OR  = 1
L_AND = 2

# Training Data
x_data = np.array([
    [0, 0, L_XOR],
    [0, 1, L_XOR],
    [1, 0, L_XOR],
    [1, 1, L_XOR],
    [0, 0, L_AND],
    [0, 1, L_AND],
    [1, 0, L_AND],
    [1, 1, L_AND],
    [0, 0, L_OR],
    [0, 1, L_OR],
    [1, 0, L_OR],
    [1, 1, L_OR],
])
y_data = np.array([
    #XOR
    [0],
    [1],
    [1],
    [0],
    #AND
    [0],
    [0],
    [0],
    [1],
    #OR
    [0],
    [1],
    [1],
    [1],
])

# Hyperparameters
n_input = 3
n_hidden = 20
n_output = 1
learning_rate = 0.1
epochs = 10000
epoch_update = epochs / 10

# TF Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# TF Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# TF Bias
b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")


# # TensorFlow setup

# In[3]:



# TF Functions
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y * tf.log(hy) - (1-Y) * tf.log(1 - hy))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

answer = tf.equal(tf.floor(hy + 0.5), Y)
accuracy = tf.reduce_mean(tf.cast(answer, "float"))


# #  Training and Evaluation

# In[4]:


with tf.Session() as session:
    session.run(init)

    for step in range(epochs):
        session.run(optimizer, feed_dict = {X: x_data, Y: y_data})

        if step % epoch_update == 0:
            print("Step: ", step)
            print("Cost: ", session.run(cost, feed_dict = {X: x_data, Y: y_data}))
            print("Accuracy: ", session.run(accuracy, feed_dict = {X: x_data, Y: y_data}))
            print(session.run([hy], feed_dict = {X: x_data, Y: y_data}))
            print("")

    print("####################\n")
    print("Output: ", session.run([hy], feed_dict = {X: x_data, Y: y_data}))
    print("Accuracy: ", session.run(accuracy, feed_dict = {X: x_data, Y: y_data}))

    print("####################\n")
    ckpt_name = "logic-gates-model.ckpt"
    print("Saving model to {0}", ckpt_name)
    saver = tf.train.Saver()
    saver.save(session, "./" + ckpt_name, step)


