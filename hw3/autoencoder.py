from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend
import tensorflow as tf
import matplotlib.pyplot as plt

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
all_label = pickle.load(open('datas/data/all_label.p','rb'))
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
unlabel = pickle.load(open('datas/data/all_unlabel.p','rb'))
unlabel = np.array(unlabel)
testdata = pickle.load(open('datas/data/test.p','rb'))
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
nb_epoch = 50

# autoencoder
input_img = Input(shape=(32,32,3))
autoencoder = Convolution2D(64, 3, 3, dim_ordering='tf', border_mode = 'same')(input_img)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(64, 3, 3, border_mode = 'same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = MaxPooling2D(pool_size=(2,2))(autoencoder)
autoencoder = Convolution2D(128, 3, 3, border_mode='same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(128, 3, 3, border_mode='same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = MaxPooling2D(pool_size=(2,2))(autoencoder)
autoencoder = Flatten()(autoencoder)
autoencoder = Dense(1024)(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Dense(512)(autoencoder)
encoded = Activation('relu')(autoencoder)

autoencoder = Dense(1024)(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Dense(128*8*8)(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Reshape(dims=(128,8,8))(autoencoder)
autoencoder = UpSampling2D(size=(2,2))(autoencoder)
autoencoder = Convolution2D(128,3,3,border_mode= 'same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(128,3,3,border_mode= 'same')(autoencoder)
autoencoder = Activation('relu')(autoencoder)
autoencoder = UpSampling2D(size=(2,2))(autoencoder)
autoencoder = Convolution2D(64, 3, 3, dim_ordering='tf', border_mode = 'same')(input_img)
autoencoder = Activation('relu')(autoencoder)
autoencoder = Convolution2D(3, 3, 3, border_mode = 'same')(autoencoder)
decoded = Activation('sigmoid')(autoencoder)

auto = Model(input_img,decoded)
auto.compile(optimizer = 'adam', loss = 'binary_crossentropy') 
unlabel = np.append(unlabel,X_train,axis=0)
callbacks = [
    EarlyStopping(monitor='val_loss', patience = 7, verbose = 0),
    ModelCheckpoint(filepath='auto_model', monitor = 'val_acc', save_best_only=True,
                    verbose=0, mode = 'max')
]
auto.fit(unlabel, unlabel,
         nb_epoch = 2,
         batch_size = 32,
         shuffle=True,
         validation_data=(X_test,X_test),
         verbose = 1,
         callbacks = callbacks)

encoded_imgs = encoder.predict(X_test[0])
decoded_imgs = decoder.predice(encoded_imgs)
plt.imshow(255.0*X_test[0])
plt.save('original.fig')
plt.imshow(255.0*decoded_imgs[0])
plt.save('decoded.fig')

#define model
'''
model = Sequential()
#zero-padding convolution2D
model.add(Convolution2D(64, 3, 3,dim_ordering='tf', border_mode = 'same', 
                        input_shape=X_train.shape[1:], W_regularizer = (l1=0.0001, l2=0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode = 'valid', W_regularizer = (l1=0.0001, l2=0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3,border_mode='same', W_regularizer=(l1=0.0001, l2=0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128,3,3, W_regularizer=(l1=0.0001, l2=0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256,3,3,border_mode='same', W_regularizer=(l1=0.0001, l2=0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(256,3,3, W_regularizer=(l1=0.0001,l2=0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('softmax'))
open('./model_self_1619.json','w').write(model.to_json())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
weights = 'weights_self_1619.h5'

Fit_X_train = X_train
Fit_Y_train = Y_train

for count in range(iteration):
#training model
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
        EarlyStopping(monitor='val_loss', patience = 6, verbose = 0),
        ModelCheckpoint(filepath=weights, monitor = 'val_acc', save_best_only=True,
                        verbose=0, save_weights_only=True, mode = 'max')
    ]
    print ("iteration : " + str(count))
    if count > 7:
        threshold = 0.98
    if count > 9 :
        threshold = 0.95
    if count > 11 :
        threshold = 0.9
    if count > 13 :
        threshold = 0.85
    print ("threshold : " + str(threshold))
    
    #train labeled data at first
    print ("training labeled data")
    data_gener.fit(Fit_X_train)
    model.fit_generator(data_gener.flow(Fit_X_train, Fit_Y_train,batch_size=batch_size),
                    samples_per_epoch=Fit_X_train.shape[0]*5,
                    nb_epoch=nb_epoch,
                    verbose = 1,
                    validation_data = (X_test,Y_test),
                    callbacks = callbacks
                    )

    #predict unlabeled data
    prediction = model.predict(unlabel,verbose = 0)
    index = np.amax(prediction, axis = 1) > threshold
    X_train = np.append(X_train, unlabel[index,:,:,:] ,axis = 0)
    temp = np.argmax(prediction, axis = 1)
    temp = np_utils.to_categorical(temp, num_class) 
    Y_train = np.append(Y_train ,temp[index,:] ,axis = 0)
    inv_index = np.invert(index)
    unlabel = unlabel[inv_index,:,:,:]
    print("labeled number"+str(len(X_train)))
    print("unlabel number"+str(len(unlabel)))
    
    print ("self learning")
    #train self-learning data
    data_gener.fit(X_train)
    model.fit_generator(data_gener.flow(X_train ,Y_train ,batch_size=batch_size),
                samples_per_epoch = X_train.shape[0]*5,
                nb_epoch = nb_epoch,
                verbose = 1,
                validation_data = (X_test,Y_test),
                callbacks = callbacks
                )

'''
if keras.backend.tensorflow_backend._SESSION:
    tf.reset_default_graph()
    keras.backend.tensorflow_backend._SESSION.close()
    keras.backend.tensorflow_backend._SESSION = None
