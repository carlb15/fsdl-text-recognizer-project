from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    model = Sequential()
    activation = 'relu'
    inputShape = input_shape
    
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        inputShape = (input_shape[0], input_shape[1], 1)
    
    # 1st conv -> activation -> pool layers
    model.add(Conv2D(20, 5, padding='same', activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
    # 2nd conv -> activation -> pool layers
    model.add(Conv2D(50, 5, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.2))
    
    # First FC -> Activation
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))
    
    # Second FC layer
    model.add(Dense(num_classes, activation='softmax'))
    
    ##### Your code above (Lab 2)

    return model