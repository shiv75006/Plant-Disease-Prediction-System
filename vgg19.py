# -*- coding: utf-8 -*-
"""PlantDisease.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12I_4Zidl6HHQI_dvtSdk3TW3KPdsyGqV
"""

import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import os

import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

train_datagen = ImageDataGenerator(zoom_range = 0.5, shear_range =0.3, horizontal_flip= True, preprocessing_function = preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train = train_datagen.flow_from_directory(directory = "/Users/shivyanshusaini/Downloads/archive (3) 2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)", target_size=(256,256), batch_size=32)

val = val_datagen.flow_from_directory(directory = "/Users/shivyanshusaini/Downloads/archive (3) 2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)", target_size=(256,256), batch_size=32)

t_img, label = train.next()

def plotImage(img_arr, label):

  for im, l in zip(img_arr, label):
    plt.figure(figsize=(5,5))
    plt.imshow(im/255)
    plt.show()

plotImage(t_img[:3], label[:3])

#building the model

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(256,256,3),include_top=False)

for layer in base_model.layers:
  layer.trainable=False

X=Flatten()(base_model.output)
X=Dense(units=2, activation='softmax')(X)

#creating the model
model = Model(base_model.input, X)

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

es=EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1)

mc=ModelCheckpoint(filepath="best_model.h5", monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1, save_best_only=True)

cb=[es,mc]

his=model.fit_generator(train, steps_per_epoch=16, epochs=50, verbose=1, callbacks=cb, validation_data=val, validation_steps=16)

h=his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c="red")
plt.title("acc vs v-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c="red")
plt.title("loss vs v-loss")
plt.show()

from keras.models import load_model

model = load_model("/content/best_model.h5")

acc = model.evaluate_generator(val)[1]
print(f"The accuracy of the model is = {acc*100}%")

img_path = '/Users/shivyanshusaini/Desktop/pdp/0a285c8b-1c31-48d4-89f2-af8b9edc36f6___RS_HL 5759.JPG'
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = model.predict(img_array)

# Assuming 'predictions' is the output of your model
predictions = model.predict(img_array)

# Get the index with the highest probability
predicted_class_index = np.argmax(predictions)

# Map the index to your class labels (replace 'class_labels' with your actual class labels)
class_labels = ["class_1", "class_2", ..., "class_38"]
predicted_class = class_labels[predicted_class_index]

# Print the result
print(f"Predicted class: {predicted_class}")

img_path = '/Users/shivyanshusaini/Desktop/pdp/0a42f1ea-8c85-458b-bc8f-c17621515a66___MD_Powd.M 0098_flipLR.JPG'
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = model.predict(img_array)

# Assuming 'predictions' is the output of your model
predictions = model.predict(img_array)

# Get the index with the highest probability
predicted_class_index = np.argmax(predictions)

# Map the index to your class labels (replace 'class_labels' with your actual class labels)
class_labels = ["class_1", "class_2", ..., "class_38"]
predicted_class = class_labels[predicted_class_index]

# Print the result
print(f"Predicted class: {predicted_class}")