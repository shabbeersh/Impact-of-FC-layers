from __future__ import print_function
import keras
from keras.utils import np_utils
import scipy
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

class CNN3_TinyImageNet:
    def __init__(self,train=True):
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('CNN3_TinyImageNet.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3),
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('softmax'))
        model.summary()
        return model

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
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        from load_images import load_images

        # Params
        loss_functions = ['categorical_crossentropy', 'squared_hinge', 'hinge']
        num_classes = 200
        batch_size = 32
        # Load images
        path = '/home/shabbeer/Desktop/Impact of FC/tiny-imagenet-200'
        X_train, y_train, X_test, y_test = load_images(path, num_classes)

        print('X_train shape', X_train.shape)
        print('X_test shape', X_test.shape)
        print('y_train shape', y_train.shape)
        print('Y_test shape', y_test.shape)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        num_samples = len(X_train)

        x_train = np.rollaxis(X_train, 1, 4)
        x_test = np.rollaxis(X_test, 1, 4)
        print('shape', x_train.shape)
        # input image dimensions
        num_channels, img_rows, img_cols = X_train.shape[1], X_train.shape[2], X_train.shape[3]

        print('rows', img_rows)
        print('cols', img_cols)

        print('channels', num_channels)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 200)
        y_test = np_utils.to_categorical(y_test, 200)

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
        #model.save('Fe_CIFAR100_CIFAR-VGG_feature_extraction_v1.h5')
        # training process in a for loop with learning rate drop every 25 epoches.

        import time
        start_time = time.time()
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

        print('Max Test accuracy:', max(history.history['val_acc']))

        return model

if __name__ == '__main__':
    from load_images import load_images

    # Params
    loss_functions = ['categorical_crossentropy', 'squared_hinge', 'hinge']
    num_classes = 200
    batch_size = 32
    nb_epoch = 30

    # Load images
    path = '/home/shabbeer/Desktop/Impact of FC/tiny-imagenet-200'
    X_train, y_train, X_test, y_test = load_images(path, num_classes)

    print('X_train shape', X_train.shape)
    print('X_test shape', X_test.shape)
    print('y_train shape', y_train.shape)
    print('Y_test shape', y_test.shape)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    num_samples = len(X_train)

    x_train = np.rollaxis(X_train, 1, 4)
    x_test = np.rollaxis(X_test, 1, 4)
    print('shape', x_train.shape)
    # input image dimensions
    num_channels, img_rows, img_cols = X_train.shape[1], X_train.shape[2], X_train.shape[3]

    print('rows', img_rows)
    print('cols', img_cols)

    print('channels', num_channels)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 200)
    y_test = np_utils.to_categorical(y_test, 200)

    model = CNN3_TinyImageNet()


    print("---  Training time in seconds ---%s " % (time.time() - start_time))
    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
print("the validation 0/1 loss is: ",loss)
