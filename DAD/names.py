import os
import pickle
import json
cropped_names = {}
for i in os.listdir('./STAIR_Lab_cropped'):
	cropped_names[i] = set()
	for j in os.listdir('./STAIR_Lab_cropped/' + i):
		cropped_names[i].add(j.split('_')[0])
#print(cropped_names.keys())

with open('valid_videos_json', 'w') as f:
	json.dump(cropped_names,f)	
