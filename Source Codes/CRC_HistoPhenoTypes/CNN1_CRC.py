from __future__ import print_function
from keras.utils import np_utils
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
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)



class CNN1_CRC:
    def __init__(self,train=True):
        self.weight_decay = 0.0005
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('CRC_CNN1.h5')


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
        model.add(Dense(4, activation='softmax', name='predictions'))
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
        PATH = "./Datasets/crchistophenotypes32_32"
        data_path = PATH
        data_dir_list = sorted_alphanumeric(os.listdir(data_path))
        print(data_dir_list)

        img_data_list = []

        for dataset in sorted_alphanumeric(data_dir_list):
            img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
            print('Loaded the images of dataset-' + '{}\n'.format(dataset))
            for img in img_list:
                #print(img)
                img_path = data_path + '/' + dataset + '/' + img
                img = image.load_img(img_path, target_size=(35, 35))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                #     x = x/255
                #print('Input image shape:', x.shape)
                img_data_list.append(x)

        img_data = np.array(img_data_list)
        # img_data = img_data.astype('float32')
        print(img_data.shape)
        img_data = np.rollaxis(img_data, 1, 0)
        print(img_data.shape)
        img_data = img_data[0]
        print(img_data.shape)
        num_of_samples = img_data.shape[0]
        print("sample", num_of_samples)
        labels = np.ones((num_of_samples,), dtype='int64')
        labels[0:7722] = 0
        labels[7722:13434] = 1
        labels[13434:20225] = 2
        labels[20225:] = 3
        names = ['epithelial', 'fibroblast', 'inflammatory', 'others']
        num_classes = 4
        Y = np_utils.to_categorical(labels, num_classes)

        x_train, x_test, y_train, y_test = train_test_split(img_data, Y, test_size=0.2, random_state=2)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)


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
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

        print('Max Test accuracy:', max(history.history['val_acc']))

        return model

if __name__ == '__main__':

    PATH = "./Datasets/crchistophenotypes32_32"
    data_path = PATH
    data_dir_list = sorted_alphanumeric(os.listdir(data_path))
    print(data_dir_list)

    img_data_list = []

    for dataset in sorted_alphanumeric(data_dir_list):
        img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            #print(img)
            img_path = data_path + '/' + dataset + '/' + img
            img = image.load_img(img_path, target_size=(35, 35))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            #     x = x/255
            #print('Input image shape:', x.shape)
            img_data_list.append(x)

    img_data = np.array(img_data_list)
    # img_data = img_data.astype('float32')
    print(img_data.shape)
    img_data = np.rollaxis(img_data, 1, 0)
    print(img_data.shape)
    img_data = img_data[0]
    print(img_data.shape)
    num_of_samples = img_data.shape[0]
    print("sample", num_of_samples)
    labels = np.ones((num_of_samples,), dtype='int64')
    labels[0:7722] = 0
    labels[7722:13434] = 1
    labels[13434:20225] = 2
    labels[20225:] = 3
    names = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    num_classes = 4
    Y = np_utils.to_categorical(labels, num_classes)
    print("x shape:",img_data.shape)
    print("y shape",Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(img_data, Y, test_size=0.2, random_state=2)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 4)
    y_test = keras.utils.to_categorical(y_test, 4)

    model = CNN1_CRC()


    print("---  Training time in seconds ---%s " % (time.time() - start_time))
    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
