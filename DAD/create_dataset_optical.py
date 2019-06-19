import os
import numpy as np
import pickle


d = np.load('dataset_x.npy',allow_pickle=True)
# a = [[1,2,3],['4','5','6'],[7,8,9]]
# b = ['4','5','60']

# if b in a:
# 	print("yes")
# else:
# 	print("no")
# for i in d:
# 	if [i[0],str(int(i[1])+2000)] in d:
# 		print("Yes",np.where(d==[i[0],str(int(i[1])+2000)]))
# 	break

# d = np.random.rand(11,10)
# print(d)
data = np.copy(d)
i=1
del_list = []
name = d[0][0]
while(i < len(d)):
	# print(d[i][0])
	# print(i)
	while(d[i][0] == name):
		i += 1	
		if i==len(d):
			break
	if i==len(d):
		del_list.extend(range(i-10,i))	
		# name = d[i][0]
		break
	del_list.extend(range(i-10,i))	
	name = d[i][0]
	
print(len(del_list))
print(data.shape)
data = np.delete(data,del_list,axis=0)
print(data.shape)
np.save('dataset_x_optical',data)


