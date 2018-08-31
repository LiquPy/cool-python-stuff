# Module for deep neural network on mnist dataset 
# Python 3

# invite all cool kids to party
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load and normalize dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

# structure of model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(125, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(125, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics= ['accuracy'])

# train model
model.fit(x_train, y_train, epochs=3)


# validations
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# save/load model
# model.save('epic_num_reader.model')
# new_model = tf.keras.model.load_model('epic_num_reader.model')

# predict
predictions = model.predict([x_test])
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
