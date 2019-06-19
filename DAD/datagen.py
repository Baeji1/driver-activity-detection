import numpy as np
import keras
import pickle



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_frames, labels, batch_size=32, dim=(299,299), n_channels=3,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_frames = list_frames
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_frames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_frames_temp = [self.list_frames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_frames_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_frames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_frames_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size),dtype=int)


        # Generate data
        for i, video in enumerate(list_frames_temp):
            # Store sample
            with open(video[0],'rb') as f1:
                temp = pickle.load(f1)
                X[i,] = temp[int(video[1])]

            # Store class
            y[i] = self.labels[video[0]].index(1)
            #y = np.array([0,0,1,0])
            #print(y.shape)

        return X,keras.utils.to_categorical(y, num_classes=self.n_classes)
