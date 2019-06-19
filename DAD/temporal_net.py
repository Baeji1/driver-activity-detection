import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,Flatten,Conv2D
from keras import backend as K
import numpy as np
from datagen import DataGenerator
from keras import metrics
from keras.callbacks import ModelCheckpoint
from datagen_optical import DataGenerator as dg


def BaejiNet_temporal_stream():
    X_input = Input(shape=(299,299,10))
    x_corrected = Conv2D(3,(3,3),padding = 'same')(X_input)
    base = InceptionResNetV2(weights='imagenet',include_top=False)(x_corrected)
    #model = Model(InputModel.input,base(InputModel.output))
    #x =  Flatten()(base)
    #x = Dense(512,activation = 'relu')(base)
    x = GlobalAveragePooling2D()(base)
    output = Dense(4,activation = 'softmax')(x)
    model = Model(inputs = X_input,outputs = output)
    return model


# def BaejiNet_spatial_stream():

#     base = InceptionResNetV2(weights='imagenet',input_shape = (299,299,3),include_top=False)
#     x = base.output
#     x = GlobalAveragePooling2D()(x)
#     # x = Dense(512,activation = 'relu')(x)
#     output = Dense(4,activation = 'softmax')(x)



#     model = Model(inputs = base.input,outputs = output)
#     return model




params = {'dim': (299,299),
          'batch_size': 10,
          'n_classes': 4,
          'n_channels': 10,
          'shuffle': True}

# Datasets
frames = np.load('dataset_x_optical.npy',allow_pickle=True)
labels = np.load('dataset_y.npy',allow_pickle=True)
fc = len(frames)
labels = labels.item()
np.random.shuffle(frames)
frames = frames[:10]
frames_train = frames[:int(fc*0.7),:]
frames_val = frames[int(fc*0.7):,:]
# Generators
# training_generator = DataGenerator(frames_train, labels, **params)
# validation_generator = DataGenerator(frames_val, labels, **params)

training_generator = dg(frames_train, labels, **params)
validation_generator = dg(frames_val, labels, **params)


model = BaejiNet_temporal_stream()
for layer in model.layers[:-3]:
   layer.trainable = False
for layer in model.layers[-3:]:
   layer.trainable = True


model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])

#print(model.summary())

filepath = "./models/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_weights_only = True ,save_best_only=True, mode='max',period = 2)
callbacks_list = [checkpoint]

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 2 ,
                    steps_per_epoch = int(np.floor(len(frames_train) / params['batch_size'])),
                    callbacks = callbacks_list,
                    verbose = 1,
                    use_multiprocessing=True,
                    workers= 6 )
