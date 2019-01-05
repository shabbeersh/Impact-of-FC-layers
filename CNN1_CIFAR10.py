from __future__ import print_function
import keras
from keras.utils import np_utils
import scipy
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from sklearn.cross_validation import train_test_split
import numpy as np
import re
import os
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import time
start_time = 0

class cifar10vgg:
    def __init__(self,train=True):
        self.weight_decay = 0.0005
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('CIFAR10_AlexNet_v2.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = 0.0005
        model.add(Conv2D(96, (5, 5), input_shape=(35, 35, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (5, 5), activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), name='block1_maxpooling2'))

        model.add(Conv2D(384, (3, 3), activation='relu', name='block1_conv3', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Conv2D(384, (3, 3), activation='relu', name='block1_conv4', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation='relu', name='block1_conv5', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4, name='dropout_2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5, name='dropout_3'))
        model.add(Dense(4096, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5, name='dropout_4'))
        model.add(Dense(4096, activation='relu', name='fc3', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5, name='dropout_5'))
        model.add(Dense(64, activation='relu', name='fc4', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5, name='dropout_6'))
        model.add(Dense(10, activation='softmax', name='predictions'))
        model.summary()
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        print(x_train.shape)
        x_train = np.array([scipy.misc.imresize(x_train[i], (35, 35, 3))
                            for i in range(0, len(x_train))]).astype('float32')

        x_test = np.array([scipy.misc.imresize(x_test[i], (35, 35, 3))
                           for i in range(0, len(x_test))]).astype('float32')

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        # training process in a for loop with learning rate drop every 25 epoches.

        import time
        start_time = time.time()
        model.save('Trained_Model_CIFAR10.h5')
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr,csv_log],verbose=2)

        print('Max Test accuracy:', max(history.history['val_acc']))

        return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print(x_train.shape)
    x_train = np.array([scipy.misc.imresize(x_train[i], (35, 35, 3))
                        for i in range(0, len(x_train))]).astype('float32')

    x_test = np.array([scipy.misc.imresize(x_test[i], (35, 35, 3))
                       for i in range(0, len(x_test))]).astype('float32')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg()


    print("---  Training time in seconds ---%s " % (time.time() - start_time))
    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
