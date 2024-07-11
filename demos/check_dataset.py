import os
import numpy as np

def generate_splits_vox(file_list):
	val_paths_loaded = [];  train_paths_loaded = []
	
	with open(file_list, 'r') as f:
		for line in f:
			path = line.strip()
			if 'test' in path:
				val_paths_loaded.append(path)
			if 'dev' in path:
				train_paths_loaded.append(path)

	print('Training', len(train_paths_loaded))	
	print('Validation', len(val_paths_loaded))
	
	# Write video paths to file for validation set
	with open('file_list_val_vox.txt', 'w') as f_val:
		for path in val_paths_loaded:
			splits = path.split('/')
			path_w = os.path.join(splits[-3], splits[-2], splits[-1])
			f_val.write(path_w + '\n')

	# Write video paths to file for training set
	with open('file_list_train_vox.txt', 'w') as f_train:
		for path in train_paths_loaded:
			# path_w = os.path.basename(path)
			splits = path.split('/')
			path_w = os.path.join(splits[-3], splits[-2], splits[-1])
			f_train.write(path_w + '\n')

	print("File lists created successfully.")

if __name__ == "__main__":

	file_list = '/fsx/rs2517/data/lists/voxceleb2_proper.txt'
	generate_splits_vox(file_list)