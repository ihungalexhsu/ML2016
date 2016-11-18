from keras.models import Sequential
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import load_model
import keras.backend.tensorflow_backend
import numpy as np
import pickle
import json
import csv
import tensorflow as tf
import sys
tf.python.control_flow_ops = tf

model = load_model(sys.argv[2])
#model = model_from_json(open('model_self_1713.json').read())
#model.load_weights('weights_self_1713.h5')

test_label = pickle.load(open(sys.argv[1]+'test.p','rb'))
test_label = np.asarray(test_label['data'])
X_te = test_label.reshape((10000,3,32,32)).transpose(0,2,3,1)
X_te.astype('float32')
X_te = X_te/255.0

predict = model.predict(X_te,verbose = 0)

with open(sys.argv[3],'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter = ',', quotechar='"')
    spamwriter.writerow(['ID']+['class'])
    for i in range(10000):
        ans = np.argmax(predict[i])
        spamwriter.writerow([i]+[ans])
if keras.backend.tensorflow_backend._SESSION:
    tf.reset_default_graph()
    keras.backend.tensorflow_backend._SESSION.close()
    keras.backend.tensorflow_backend._SESSION = None

