import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = 'E:/GOOGLE DRIVE/Computer Science/PythonFiles/cool-python-stuff/cool-python-stuff/datasets/PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize all images
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# shuffle
random.shuffle(training_data)

x = []
y = []
for feature, label in training_data:
    x.append(feature)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open('datasets/PetImages/cats&dogs_x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('datasets/PetImages/cats&dogs_y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
