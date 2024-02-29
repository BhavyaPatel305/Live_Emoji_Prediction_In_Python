# Import Modules
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

# Imports for training the model
from keras.layers import Input, Dense
# Also, we use functional API from keras.model
from keras.models import Model

is_init = False

# Size of data
size = -1

# For y, we cannot pass strings like hello, goodluck, nope (hello.npy, goodluck.npy, nope.npy) to the model
# So we need to convert them to numbers
# To do that we create a dictionary
# Eg: hello -> 0, goodluck -> 1, nope -> 2
label = []
dictionary = {}
c = 0

# Search for all the .npy files in the directory, which have data in them
for i in os.listdir():
    # Only interested in .npy files
    if i.split(".")[-1] == "npy":
        if not(is_init):
            is_init = True
            # Input Data: Load the .npy file
            X = np.load(i)
            
            # Interested in 0th part, meaning no.of rows we have
            size = X.shape[0]
            
            # Label Associated to the Data
            # Eg: goodluck.npy data file
            # So we are intrested in 0th element, that is, goodluck
            y = np.array([i.split(".")[0]]*size).reshape(-1,1) # Same size as X
        else:
            # Concatenate X and the new data we are getting
            X = np.concatenate((X, np.load(i)))
            # Concatenate y and the new label we are getting
            y = np.concatenate((y, np.array([i.split(".")[0]]*size).reshape(-1,1)))
        
        # Add the label to the label array
        label.append(i.split(".")[0])
        # Add the label to the dictionary and attach value of c to it
        dictionary[i.split(".")[0]] = c
        # Increment value of variable c
        c += 1


# Convert y to integer
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i,0]]
# Integers are stored as strings, so convert them to integers
y = np.array(y,dtype="int32")   

#print(y): Illustration Purpose
# Here what we are trying to do is:
# In dictionary, say our prediction is hello, which is 0
# index = 0: array will be like [1, 0, 0]
# index = 1: array will be like [0, 1, 0]
# index = 2: array will be like [0, 0, 1]

# Otherwise problem, is 
# since hello = 0, nope = 2, there is no way to compare hello and nope
# and comparing 0 and 2 might give nope a higher priority than hello, which is wrong
# so to solve the problem we use to_categorical

y = to_categorical(y)
#print(y): Illustration Purpose

# Now with the previous version of the code,
# the problem is that first our model is learning hello, hello, hello, ...
# then it learns goodluck, goodluck, goodluck, ...
# then it learns nope, nope, nope, ...
# which is not good
# Our model could do better predictions if our data is shuffled
X_new = X.copy()
y_new = y.copy()
counter = 0

# What arange function returns:
# Eg: np.arange(5) -> [0, 1, 2, 3, 4]
cnt = np.arange(X.shape[0]) # X.shape[0] means no.of rows in X
# Shuffling the data
np.random.shuffle(cnt)

# Iterating the loop to fill shuffled data in X_new and y_new
# i will have the shuffled values
for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1
# print(y): testing purpose
# print(y_new): testing purpose

# Building Model

# Input Layer
# It only needs to deal with no.of columns we have
ip = Input(shape=(X.shape[1]))

# Middle Layers
# 512 neurons
m = Dense(512, activation="relu")(ip) # Connect with previous layer(ip)
# 256 neurons
m = Dense(256, activation="relu")(m) # Connect with previous layer(m)

# Output Layer
# In output, we need 3 neurons: like [0, 0, 1]
# neurons making predictions associate with the 0th, 1st, and 2nd index
# so we pass: y.shape[1]
# activation function: softmax
# Because we want every neuron to have a probablity associated with it
# like in [0, 0, 1], probality of being in 0th class/index, probablity of being in 1st class/index, probablity of being in 2nd class/index
# Sum of all these probablities = 1

op = Dense(y.shape[1], activation="softmax")(m) # Connect with previous middle layer(m)

# Create the model
model = Model(inputs=ip, outputs=op)

# Compile the model
# loss = "categorical_crossentropy" as we have categorical data
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Fit the model
model.fit(X, y, epochs=50)

# After shuffling the data, now the model should be accurate to predict

# Save the model
model.save("model.h5")
np.save("labels.npy", np.array(label))