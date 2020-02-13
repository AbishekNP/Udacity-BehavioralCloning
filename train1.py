import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import shuffle



samples = []
with open('../car_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                
                for line in batch_sample:
                    is_first_row=True
                    if is_first_row:
                        is_first_row = False
                        continue
            
                    filename = './IMG/'+line[0].split('/')[-1]
                    center_image_name = line[0]
                    left_image_name = line[1]
                    right_image_name = line[2]
                    image1 = cv2.imread(center_image_name)
                    image2 = cv2.imread(left_image_name)
                    image3 = cv2.imread(right_image_name)
                    images += [image1, image2, image3]
                    correction = 0.2
                    steering_center = float(line[3])
                    steering_left = float(steering_center + correction)
                    steering_right = float(steering_center - correction)
                    measurements += [steering_center, steering_left, steering_right]


                    X_train = np.asarray(images)
                    y_train = np.asarray(measurements)
                    yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size=(5,5), subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(filters=36, kernel_size=(5,5), subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.2))
model.add(Conv2D(filters=48, kernel_size=(5,5), subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(ELU())
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(ELU())


model.add(Flatten())

model.add(Dense(units=1164))
model.add(ELU())
model.add(Dense(units=100))
model.add(ELU())
model.add(Dropout(0.2))
model.add(Dense(units=50))
model.add(ELU())
model.add(Dense(units=10))
model.add(Dense(units=1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=10, verbose=1)

model.save('model.h5')

    






 


                                 
                                 
                                 
