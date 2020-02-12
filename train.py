import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import math



lines = []
with open('../car_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


images = []
measurements = []
is_first_row=True
for line in samples:
    if is_first_row:
        is_first_row = False
        continue
   
    source_path = line[0]
    filename = source_path.split('/')[-1]
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

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.asarray(augmented_images)
y_train = np.asarray(augmented_measurements)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
datagen= ImageDataGenerator()
train_generator = model.fit_generator(datagen.flow(X_train, y_train), batch_size=batch_size)
validation_generator = model.fit_generator(datagen.flow(X_test, y_test), steps_per_epoch=len(X_train)//batch_size,epochs=5)




model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size=(5,5), subsample=(2,2), activation="relu"))
model.add(Conv2D(filters=36, kernel_size=(5,5), subsample=(2,2), activation="relu"))
model.add(Conv2D(filters=48, kernel_size=(5,5), subsample=(2,2), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))

model.add(Flatten())

model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=50, activation="relu"))
model.add(Dense(units=10))
model.add(Dense(units=1))

model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

model.save('model.h5')

    






 


                                 
                                 
                                 
