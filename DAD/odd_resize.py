import cv2
import numpy as np
import os
import pickle

f1 = open('valid_videos','rb')
d = pickle.load(f1)
f1.close()

for i in os.listdir('./STAIR_Lab'):
	for j in os.listdir('./STAIR_Lab/' + i):
		#print(j)
		if j[:-4] in d[i]:
			#print("lol") 
			cap = cv2.VideoCapture('./STAIR_Lab/' + i + '/' + j)
			 
			#fourcc = cv2.VideoWriter_fourcc(*'XVID')
			#out = cv2.VideoWriter('./STAIR_Lab_299/' + i + '/' + j,fourcc, 30, (299,299))
			video = [] 
			while True:
				ret, frame = cap.read()
				if ret == True:
					b = cv2.resize(frame,(299,299),fx=0,fy=0) #interpolation = cv2.INTER_CUBIC)
					#print(b.shape)
					video.append(b)
				else:
					break 
			f1 = open('./STAIR_Lab_299/' + i + '/' + j[:-4],'wb')
			pickle.dump(video,f1)
			f1.close()  
			cap.release()
			#out.release()
			cv2.destroyAllWindows()
