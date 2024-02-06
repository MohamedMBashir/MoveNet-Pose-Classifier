import tensorflow as tf
from tensorflow import keras
from movenet import landmarks_to_embedding




class_names = ['Down', 'Neutral', 'Up']

# Define the model
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)
# model.summary()

model.load_weights('./models/squat_pose_classification_model_weights.best.hdf5')