
import os
import pickle
import numpy as np

path = './STAIR_Lab_299/'
x = []
y = []
target = {'Drinking': [1,0,0,0], 'Eating_snack':[0,1,0,0], 'Smoking':[0,0,1,0], 'Telephoning':[0,0,0,1]}
z = 0
# for i in os.listdir(path):
# 	for j in os.listdir(path + i + '/'):
# 		z += 1
# 		with open(path + i + '/' + j,'rb') as f1:
# 			data = pickle.load(f1) 
# 		x.extend(data)
# 		y.extend([target[i] for k in range(len(data))])
# 		if z%10 == 0:
# 			print('Finished ' + str(z) + ' videos')
# 		# print('x',x,np.array(x).shape)
# 		# print('y',y,np.array(y).shape)
# 	print('Done with ' + i)


# with open('Dataset','wb') as f1:
# 	pickle.dump(x,f1)
# 	pickle.dump(y,f1)

# print('Completed loading ' + str(z) + ' videos')

dataset_x = np.array([[None,None]])
dataset_y = {}
z = 0
for (root,dirs,files) in os.walk(path,topdown=True):
	if len(files) > 0:
		print(root)
		s = target[root[len(path):]]
		for i in files:
			z += 1
			if z%10 == 0:
				print(z)
			dataset_y[root + '/' + i] = s
			with open(root + '/' + i, 'rb')as f1:
				d = pickle.load(f1)
				d = len(d)
			for j in range(0,d):
				dataset_x = np.append(dataset_x,np.array([[root + '/' + i,j]]),axis=0)

dataset_x = np.delete(dataset_x,(0),axis=0)

np.save('dataset_x',dataset_x)
np.save('dataset_y',dataset_y)
print('Loaded ' + z + ' files')