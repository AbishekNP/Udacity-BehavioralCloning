import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense


lines = []
with open('../car_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
is_first_row=True
for line in lines:
    if is_first_row:
        is_first_row = False
        continue
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../car_data/driving_log.csv'
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
#map(float, measurements)



X_train = np.asarray(images)
y_train = np.asarray(measurements)

print(X_train[1].shape)
#model = Sequential()
#model.add(Flatten(input_shape=(160,320,3)))
#model.add(Dense(1))

#model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

#model.save('model.h5')

    






 


                                 
                                 
                                 
