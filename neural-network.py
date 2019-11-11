import tensorflow as tf
tf.add(1, 2).numpy()

hello = tf.constant('Hello, TensorFlow!')
hello.numpy()

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
