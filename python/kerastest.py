#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K

from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

from keras.wrappers.scikit_learn import KerasClassifier
from functools import partial

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

input_shape = None
num_classes = None
def build_model1():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    return model

def build_model2():
    model = Sequential()
    model.add( Conv2D(28, kernel_size=(4,4), strides=(2,2), activation='relu',input_shape=input_shape) )
    model.add( AveragePooling2D(pool_size=(2,2), strides=(2,2)) )
    model.add( Conv2D(64, kernel_size=(4,4), strides=(2,2), activation='relu') )
    model.add( AveragePooling2D(pool_size=(2,2), strides=(2,2)) )
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
    return model

def _main():

    batch_size = 32
    epochs = 1
    num_predictions = 20
#    data_augmentation = False
    img_rows, img_cols = 28, 28

#    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    global input_shape, num_classes
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X = np.vstack((x_train,x_test))
    y = np.vstack((y_train,y_test))

    input_shape = x_train.shape[1:]

    history = AccuracyHistory()

#    if data_augmentation:
#        datagen = ImageDataGenerator(
#        featurewise_center=False,  # set input mean to 0 over the dataset
#        samplewise_center=False,  # set each sample mean to 0
#        featurewise_std_normalization=False,  # divide inputs by std of the dataset
#        samplewise_std_normalization=False,  # divide each input by its std
#        zca_whitening=False,  # apply ZCA whitening
#        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#        horizontal_flip=True,  # randomly flip images
#        vertical_flip=False)  # randomly flip images
#
#        # Compute quantities required for feature-wise normalization
#        # (std, mean, and principal components if ZCA whitening is applied).
#        datagen.fit(x_train)
#
#    # Fit the model on the batches generated by datagen.flow().
#        model1.fit_generator(datagen.flow(x_train, y_train,
#                                         batch_size=batch_size),
#                            epochs=epochs,
#                            validation_data=(x_test, y_test),
#                            workers=4,
#                            callbacks=[history])
#
#
#    else:


    model1 = KerasClassifier( build_fn=build_model1,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1 )
#                              validation_data=(x_test, y_test),
#                              callbacks=[history])


    model2 = KerasClassifier( build_fn=build_model2,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1 )
#                              validation_data=(x_test, y_test),
#                              callbacks=[history])


    stacked_model = StackingClassifier( classifiers=[model1,model2],
                                        meta_classifier=LogisticRegression( penalty='l2' ),
                                        use_probas=True )

    for m,name in zip((model1, model2, stacked_model), ('CNN 1', 'CNN 2', 'Stacked Model')):
        kfold = KFold(n_splits=10, shuffle=True)
        scores = cross_val_score( m, X, y.argmax(axis=1), cv=kfold, scoring='accuracy' )
        print( '{}: {}'.format(name,scores.mean()) )


#    score = model.evaluate(x_test, y_test, verbose=0)
#    print('Test loss:', score[0])
#    print('Test accuracy:', score[1])

#    plt.plot(range(1,epochs+1), history.acc)
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy')
#    plt.show()


if __name__ == '__main__':
    import sys;
    sys.exit( _main() )

