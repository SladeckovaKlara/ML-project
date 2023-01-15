import numpy as np
import sys  
import cv2
import os
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D
from keras import backend as K
from imageio.v2 import imread
from PIL import Image as Image
import tensorflow as tf
import random

def load_images_from_folder(folder):
    images = []
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        
        if img is not None:
          img = Image.fromarray(img)
          new_img = np.resize(img, (96, 96, 3))
          vc = new_img.flatten()
          images.append(np.array(vc))
     
    
    # fig, axarr = plt.subplots(10, 1, figsize=(12, 12))
    # for i in range(10):
       # axarr[i].imshow(images[i], 'gray')
       # axarr[i].axis('off')

    # plt.show()
    
    return images

def distribute_train_test_data(images):
  train, test = [] , []
  
  count = 0
  for image in images:
    if (count%7 == 0):
      test.append(image)
    else:
      train.append(image)
    count = count+1
      
  return np.array(train), np.array(test)

def shuffle(X, y):
  shuf = np.arange(0, len(X))
  np.random.shuffle(shuf)
  
  new_X, new_y = [], []
  for i in range (len(X)):
    new_X.append(X[shuf[i]])
    new_y.append(y[shuf[i]])
    
  return np.array(new_X), np.array(new_y)

def generuj_sum(X_test):
  for i in range (len(X_test)):
    rand = random.randint(-25, 25)
    X[i] = X[i]-rand

def fit(X_train, y_train, X_test, y_test):
  # X_train = (X_train-np.mean(X_train)) / np.std(X_train)  
  # X_test = (X_test-np.mean(X_train)) / np.std(X_train)
  img_rows = 96
  img_cols = 96

  if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
  else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
  print(np.shape(X_train))
  
  cnn = Sequential()

  cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding = 'valid'))
  cnn.add(Conv2D(64, (5, 5), activation='relu'))
  cnn.add(Conv2D(128, (3, 3), activation='relu', padding = 'valid'))
  cnn.add(MaxPooling2D(pool_size=(2, 2)))
  cnn.add(Conv2D(32, (3, 3), activation='relu', padding = 'valid'))
  cnn.add(Conv2D(64, (5, 5), activation='relu'))
  cnn.add(Conv2D(128, (3, 3), activation='relu', padding = 'valid'))
  cnn.add(MaxPooling2D(pool_size=(2, 2)))
  cnn.add(Flatten())

  cnn.add(Dense(256, activation='relu'))
  cnn.add(Dense(256, activation='relu'))
  cnn.add(Dense(5, activation='softmax'))

  cnn.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])

  history = cnn.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=30, batch_size=256, 
  validation_split=0.1, verbose=True)
  
  plt.figure()
  plt.plot(history.history['loss'], label='training loss')
  plt.plot(history.history['val_loss'], label='validation loss')
  plt.legend(loc='best')

  plt.figure()
  plt.plot(history.history['accuracy'], label='train accuracy')
  plt.plot(history.history['val_accuracy'], label='validation accuracy')
  plt.legend(loc='best')
  plt.show()
  
  
  train_score = cnn.evaluate(X_train, tf.keras.utils.to_categorical(y_train))
  test_score = cnn.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
  
  return train_score, test_score

front_train, front_test = distribute_train_test_data( load_images_from_folder(sys.argv[1]))
left_train, left_test = distribute_train_test_data( load_images_from_folder(sys.argv[2]))
lower_train, lower_test = distribute_train_test_data( load_images_from_folder(sys.argv[3]))
right_train, right_test = distribute_train_test_data( load_images_from_folder(sys.argv[4]))
upper_train, upper_test = distribute_train_test_data( load_images_from_folder(sys.argv[5]))

X_train = np.concatenate((front_train, left_train, lower_train, right_train, upper_train))
X_test = np.concatenate((front_test, left_test, lower_test, right_test, upper_test))

y_train = np.concatenate((np.full((len(front_train)), 0), np.full((len(left_train)), 1), np.full((len(lower_train)), 2), np.full((len(right_train)), 3), np.full((len(upper_train)), 4)))

y_test = np.concatenate((np.full((len(front_test)), 0), np.full((len(left_test)), 1), np.full((len(lower_test)), 2), np.full((len(right_test)), 3), np.full((len(upper_test)), 4)))

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

#X_test = generuj_sum(X_test)

train_score, test_score = fit(X_train, y_train, X_test, y_test)

print("\n\ntrain loss: {} | train acc: {}\n".format(train_score[0], train_score[1]))
print("\n\ntest loss: {} | test acc: {}".format(test_score[0], test_score[1]))
