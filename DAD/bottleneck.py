import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from datagen_train import DataGenerator as DataGenerator_train
from datagen_predict import DataGenerator as DataGenerator_predict
from keras import metrics
from keras.callbacks import ModelCheckpoint
import time
import pickle



def BaejiNet_spatial_stream():
	base = InceptionResNetV2(weights='imagenet',input_shape = (299,299,3),include_top=False)
	x = base.output
	output = GlobalAveragePooling2D()(x)
    #x = Dense(512,activation = 'relu')(x)
    #output = Dense(4,activation = 'softmax')(x)



	model = Model(inputs = base.input,outputs = output)
	return model




params = {'dim': (299,299),
          'batch_size': 10,
          'n_classes': 4,
          'n_channels': 3,
          'shuffle': False}

# Datasets
frames = np.load('dataset_x.npy',allow_pickle=True)
fc = len(frames)
#np.random.shuffle(frames)

# Generators
predict_generator = DataGenerator_predict(frames, **params)

model = BaejiNet_spatial_stream()


model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])

#print(model.summary())


start = time.time()
out = model.predict_generator(generator=predict_generator)
out = np.concatenate((frames,out),axis=1)
np.save('bottleneck',out)
print(time.time()-start)
