from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
from keras.models import load_model #ADDED
import cv2 #ADDED
import numpy as np
import argparse
import os
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="Dataset/test_set",
	help="path to directory containing the '3scenes' dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...") #ADDED
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

for imagePath in imagePaths:
	image = Image.open(imagePath)
	image = np.array(image.resize((32, 32))) / 255.0
	data.append(image)

	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

print("Labels ===", np.array(labels), "\n") #ADDED

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25)

print("Labels ===", np.array(labels), "\n") #ADDED

model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(2))# ===Number of perameters/label types 2 for Cat and Dog, 3 for Forest/Coast/Highway
model.add(Activation("softmax"))

trainY = tf.keras.utils.to_categorical(trainY, 2)
testY = tf.keras.utils.to_categorical(testY, 2)

print("[INFO] training network...")
opt = Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=25, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

#model.save('model.h5')

print ("#Predictions = ", len(predictions))

image = Image.open("sample3.png")
image = np.array(image.resize((32, 32))) / 255.0
image = np.expand_dims(image, axis=0) #adds a 4th dimension, adding 1 in beginning
imagePrediction = model.predict(image)
print(imagePrediction)

# image = Image.open("sample3.jpg")
# image = np.array(image.resize((32, 32))) / 255.0
# image = np.expand_dims(image, axis=0) #adds a 4th dimension, adding 1 in beginning
# imagePrediction = model.predict(image)
# print(imagePrediction)
