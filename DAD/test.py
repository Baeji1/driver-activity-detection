import numpy as np
np.set_printoptions(threshold=np.inf)


d = np.load('dataset_x_optical.npy', allow_pickle = True)
dd = np.load('dataset_x.npy',allow_pickle = True)
for i in d:
	if i[0] == './STAIR_Lab_299/Telephoning/a032-1050C':
		print(i)

for i in dd:
	if i[0] == './STAIR_Lab_299/Telephoning/a032-1050C':
		print(i)
