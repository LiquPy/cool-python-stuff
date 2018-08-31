import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# load data from pickled datasets
pickle_in = open('datasets/PetImages/cats&dogs_x.pickle', 'rb')
x = pickle.load(pickle_in)
pickle_in.close

pickle_in = open('datasets/PetImages/cats&dogs_y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close

x = x/255

model = Sequential()

# layer 1 
model.add(Conv2D(64, (3,3), input_shape= x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 2
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 3
model.add(Flatten())
model.add(Dense(64))

# layer 4
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=32, validation_split=0.1, epochs=3)

