from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend
import tensorflow as tf
import sys

tf.python.control_flow_ops=tf
sess = tf.Session()
K.set_session(sess)

import numpy as np
import pickle

iteration = 6
threshold = 0.995
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
print (unlabel.shape)
#format data
X_train = X_train.reshape(int(num_data*num_class*(1-val_split)),img_channels,img_rows,img_cols).transpose(0,2,3,1)
X_test = X_test.reshape(int(num_data*num_class*val_split),img_channels,img_rows, img_cols).transpose(0,2,3,1)
Y_train = np_utils.to_categorical(y_train.astype(int), num_class)
Y_test = np_utils.to_categorical(y_test.astype(int), num_class)

batch_size = 32
nb_classes = 10
nb_epoch = 40

#define model
model = Sequential()
model.add(Convolution2D(64, 3, 3,dim_ordering='tf', border_mode = 'same', 
                        input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode = 'valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(256,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('softmax'))
#open('./model_self_1717.json','w').write(model.to_json())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#weights = 'weights_self_1717.h5'

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
unlabel = unlabel.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.
unlabel = unlabel / 255.
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
        EarlyStopping(monitor='val_loss', patience = 5, verbose = 1),
        ModelCheckpoint(filepath=sys.argv[2], monitor = 'val_acc', save_best_only=True,
                        verbose=1, mode = 'max')
    ]
    print ("iteration : " + str(count))
    if count > 4:
        threshold = 0.98
    if count > 5 :
        threshold = 0.95
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
    print ("iter : "+str(count)) 
    print ("self learning")
    #train self-learning data
    data_gener.fit(X_train)
    model.fit_generator(data_gener.flow(X_train ,Y_train ,batch_size=batch_size),
                samples_per_epoch = X_train.shape[0]*3,
                nb_epoch = nb_epoch,
                verbose = 1,
                validation_data = (X_test,Y_test),
                callbacks = callbacks
                )


if keras.backend.tensorflow_backend._SESSION:
    tf.reset_default_graph()
    keras.backend.tensorflow_backend._SESSION.close()
    keras.backend.tensorflow_backend._SESSION = None
