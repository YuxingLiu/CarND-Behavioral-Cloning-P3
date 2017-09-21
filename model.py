"""
Steering angle prediction model
"""
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


path = 'data_track1'
samples = []
with open(path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # Record the path of center camera images and steer angle
        center_path = path + '/IMG/' + line[0].split('\\')[-1]
        center_angle = float(line[3])
        # To flip images within the generator function, add flag_flip as the third entry
        # flag_flip=1 for original image, flag_flip=0 for flipped image
        samples.append((center_path, center_angle, 1))

        # Create adjusted steering measurements for the side camera images
        correction = 0.1  # this is a parameter to tune
        left_path = path + '/IMG/' + line[1].split('\\')[-1]
        left_angle = center_angle + correction
        samples.append((left_path, left_angle, 1))
        right_path = path + '/IMG/' + line[2].split('\\')[-1]
        right_angle = center_angle - correction
        samples.append((right_path, right_angle, 1))

        # Flip images horizontally for data augmentation
        samples.append((center_path, -center_angle, -1))
        samples.append((left_path, -left_angle, -1))
        samples.append((right_path, -right_angle, -1))


# Split the dataset, use 80% for training and 20% for validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(samples))
print(len(train_samples))
print(len(validation_samples))


# Use a generator to load data and preprocess it on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = mpimg.imread(batch_sample[0])
                angle = float(batch_sample[1])
                if batch_sample[2] > 0:
                    images.append(image)                # original image
                else:
                    images.append(cv2.flip(image, 1))   # flipped image
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Define network architecture using Keras
model = Sequential()
# Normalization
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
# Trim image to only see section with road
model.add(Cropping2D(cropping=((60, 20), (0, 0))))

# Layer 1: 5x5 Convolutional. Input = 80x320x3. Output = 76x316x16.
model.add(Conv2D(16, 5, 5, activation='relu'))
# 2x2 Pooling. Output = 38x158x16.
model.add(MaxPooling2D())

# Layer 2: 5x5 Convolutional. Output = 34x154x24.
model.add(Conv2D(24, 5, 5, activation='relu'))
# 2x2 Pooling. Output = 17x77x24.
model.add(MaxPooling2D())

# Layer 3: 5x5 Convolutional. Output = 13x73x32.
model.add(Conv2D(32, 5, 5, activation='relu'))
# 2x2 Pooling. Output = 6x36x32.
model.add(MaxPooling2D())

# Layer 4: 3x3 Convolutional. Output = 4x34x48.
model.add(Conv2D(48, 3, 3, activation='relu'))
# 2x2 Pooling. Output = 2x17x48.
model.add(MaxPooling2D())

# Layer 5: 2x2 Convolutional. Output = 1x16x64.
model.add(Conv2D(64, 2, 2, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))

# Layer 6: Fully Connected. Input = 1024. Output = 128.
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Layer 7: Fully Connected. Output = 16.
model.add(Dense(16, activation='relu'))

# Layer 8: Fully Connected. Output = steer angle.
model.add(Dense(1))


# Train the model on data generated batch-by-batch by a Python generator
model.compile(loss='mse', optimizer='adam')
hist = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), nb_epoch=10,
                           validation_data=validation_generator, nb_val_samples=len(validation_samples))
model.save('model.h5')


# Plot the training and validation loss for each epoch
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()