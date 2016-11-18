from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend
import tensorflow as tf
import sys
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

tf.python.control_flow_ops=tf
sess = tf.Session()
K.set_session(sess)

import numpy as np
import pickle

seed = 5
np.random.seed(seed)
img_rows, img_cols = 32,32
img_channels = 3
num_class=10
num_data=500
val_split = 0.1

#read label data
all_label = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
all_label = np.array(all_label)
X_train = np.empty((0,img_channels*img_rows*img_cols))
X_test = np.empty((0,img_channels*img_rows*img_cols))
y_train = np.empty((0,1),dtype='int')
y_test = np.empty((0,1),dtype='int')
for i in range(num_class):
    temp = all_label[i]
    np.random.shuffle(temp)
    X_test = np.append(X_test,temp[0:50],axis = 0)
    X_train = np.append(X_train,temp[50:],axis = 0)
    temp = np.zeros((num_data,1))
    temp.fill(int(i))
    y_test = np.append(y_test,temp[0:50],axis = 0)
    y_train = np.append(y_train,temp[50:],axis = 0)

#read unlabeled data
unlabel = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
unlabel = np.array(unlabel)
testdata = pickle.load(open(sys.argv[1]+'test.p','rb'))
testdata = np.asarray(testdata['data'])
testdata = testdata.reshape((10000,3,32,32)).transpose(0,2,3,1)
unlabel = unlabel.reshape((45000,img_channels,img_rows,img_cols)).transpose(0,2,3,1)
unlabel = np.append(unlabel,testdata,axis=0)
#format data
X_train = X_train.reshape(int(num_data*num_class*(1-val_split)),img_channels,img_rows,img_cols).transpose(0,2,3,1)
X_test = X_test.reshape(int(num_data*num_class*val_split),img_channels,img_rows, img_cols).transpose(0,2,3,1)
Y_train = np_utils.to_categorical(y_train.astype(int), num_class)
Y_test = np_utils.to_categorical(y_test.astype(int), num_class)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
unlabel = unlabel.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.
unlabel = unlabel / 255.

batch_size = 32
nb_classes = 10
nb_epoch = 100

# autoencoder
input_img = Input(shape=(32,32,3))
autoencoder = Convolution2D(64, 3, 3, dim_ordering='tf', border_mode = 'same', activity_regularizer=regularizers.activity_l2(10e-5))(input_img)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(64, 3, 3, border_mode = 'same', activity_regularizer=regularizers.activity_l2(10e-5))(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = MaxPooling2D(pool_size=(2,2))(autoencoder)
autoencoder = Convolution2D(128, 3, 3, border_mode='same',activity_regularizer=regularizers.activity_l2(10e-5))(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(128, 3, 3, border_mode='same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = MaxPooling2D(pool_size=(2,2))(autoencoder)
autoencoder = Flatten()(autoencoder)
autoencoder = Dense(1024)(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Dense(512)(autoencoder)
encoder = Activation('relu')(autoencoder)

autoencoder = Dense(1024)(encoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Dense(128*8*8)(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Reshape((128,8,8),input_shape=(128*8*8,))(autoencoder)
autoencoder = UpSampling2D(size=(2,2))(autoencoder)
autoencoder = Convolution2D(128,3,3,border_mode= 'same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(128,3,3,border_mode= 'same',activity_regularizer=regularizers.activity_l2(10e-5))(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = UpSampling2D(size=(2,2))(autoencoder)
autoencoder = Convolution2D(64, 3, 3, dim_ordering='tf', border_mode = 'same',activity_regularizer=regularizers.activity_l2(10e-5))(input_img)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(3, 3, 3, border_mode = 'same',activity_regularizer=regularizers.activity_l2(10e-5))(autoencoder)
decoder = Activation('sigmoid')(autoencoder)

auto = Model(input_img,decoder)
enc = Model(input_img,encoder)
auto.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy']) 
unlabel = np.append(unlabel,X_train,axis=0)
callbacks = [
    EarlyStopping(monitor='val_loss', patience = 20, verbose = 0),
    ModelCheckpoint(filepath='autoencoder_model', monitor = 'val_loss', save_best_only=True,
                    verbose=0, mode = 'auto')
]
auto.fit(unlabel, unlabel,
         nb_epoch = 40,
         batch_size = 32,
         shuffle=True,
         validation_data=(X_test,X_test),
         verbose = 1,
         callbacks = callbacks)

# using encoder as pretrain
x = Dense(2048)(encoder)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(1024)(encoder)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512)(encoder)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_class)(x)
x = Activation('softmax')(x)

model = Model(input_img, x) 
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()

data_gener = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False)

callbacks = [
    EarlyStopping(monitor='val_loss', patience = 30, verbose = 0),
    ModelCheckpoint(filepath=sys.argv[2], monitor = 'val_acc', save_best_only=True,
                    verbose=0, mode = 'max')
]
data_gener.fit(X_train)
model.fit_generator(data_gener.flow(X_train ,Y_train ,batch_size=batch_size),
                    samples_per_epoch = X_train.shape[0]*5,
                    nb_epoch = nb_epoch,
                    verbose = 1,
                    validation_data = (X_test,Y_test),
                    callbacks = callbacks
                    )

if keras.backend.tensorflow_backend._SESSION:
    tf.reset_default_graph()
    keras.backend.tensorflow_backend._SESSION.close()
    keras.backend.tensorflow_backend._SESSION = None
