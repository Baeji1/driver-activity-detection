import keras
import pickle
import numpy as np

def data_generator(list_frames,labels,batch_size=32, dim=(299,299), n_channels=3,n_classes=4, shuffle=True):
	index = 1
	while(index):
		indexes = np.arange(len(list_frames))
		if shuffle == True:
			np.random.shuffle(indexes)

		batch_indexes = indexes[index*batch_size:(index+1)*batch_size]
		list_frames_temp = [list_frames[k] for k in batch_indexes]

		X = np.empty((batch_size,*dim,n_channels))
		y = np.empty((batch_size,n_classes))

		for i, video in enumerate(list_frames_temp):
			with open(video[0],'rb') as f1:
				temp = pickle.load(f1)
				X[i,] = temp[int(video[1])]
			y[i] = labels[video[0]]

		yield X,y
