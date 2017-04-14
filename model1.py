import os
import csv
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda

from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint

from keras.models import model_from_json

#1. Create generator

# Images filepaths for generator
samples = []


def add_to_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


samples = add_to_samples('./data/driving_log.csv', samples)

samples = add_to_samples('./data/simulator_training_data/driving_log.csv', samples)  # header already removed

# Remove header, Udacity data comes with a header, simulator stores data without a header
samples = samples[1:]

print("Samples: ", len(samples))

# Split samples into training and validation sets to reduce overfitting
#10 perdent of the data points selected as validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []
            correction = 0.2 #Correction to be applied to the left and right images
            for batch_sample in batch_samples:
                name = './data/' + batch_sample[0]
                center_image = mpimg.imread(name)
                steering_angle = float(batch_sample[3])
                left_name = './data/' + batch_sample[1].lstrip()  # had to strip the leading white space in the left image value from CSV
                left_image = mpimg.imread(left_name)
                right_name = './data/' + batch_sample[2].lstrip()  # had to strip the leading white space in the right image value from CSV
                right_image = mpimg.imread(right_name)
                images.append(center_image)
                steering_angles.append(steering_angle)
                images.append(np.fliplr(center_image)) #flipped center image
                steering_angles.append(-1.0 * steering_angle)
                images.append(left_image)
                steering_angles.append(steering_angle + correction)
                images.append(np.fliplr(left_image)) #flipped left image
                steering_angles.append(-1.0 * (steering_angle + correction))
                images.append(right_image)
                steering_angles.append(steering_angle - correction)
                images.append(np.fliplr(right_image)) #flipped right image
                steering_angles.append(-1.0 * (steering_angle -correction))

            X_train = np.array(images)
            y_train = np.array(steering_angles)

            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#2. Preprocess Data

def resize_comma(image):
    import tensorflow as tf  # This import is required to prevent pre-processing in drive.py, had to update tf in the starter kit environment
    return tf.image.resize_images(image,( 40, 160))


#3. Model

# Model adapted from Comma.ai model

model = Sequential()

# Crop 70 pixels from the top of the image to eliminate objects above the driving horizon and 25 from the bottom to eliminate the hood
model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                     dim_ordering='tf',  # default
                     input_shape=(160, 320, 3)))

# Resize the data
model.add(Lambda(resize_comma))

# Normalise and mean center by subtracting 0.5  the images as in the course lecture
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Conv layer 1
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())

# Conv layer 2
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())

# Conv layer 3
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))

#Flatten, dropout to prevent overfitting
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())

# Fully connected layer 1
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())

# Fully connected layer 2
model.add(Dense(50))
model.add(ELU())

#Final layer only one neuron as the steering wheel angle is continous
model.add(Dense(1))

#Adam optimizer to manage learning rate
adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

print("Model summary:\n", model.summary())

# 4. Training
batch_size = 32
nb_epoch = 2

# weights after every epoch
checkpointer = ModelCheckpoint(filepath="./tmp/v2-weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1,
                               save_best_only=False)

# Train model using generator
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 3 * 2, # multiplier for flipped and left right images
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=nb_epoch,
                    callbacks=[checkpointer])

# 5. Final Model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save("model.h5")
print("Saved model to disk")