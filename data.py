import os
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import tarfile
import shutil
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import numpy as np
dataset_folder_path = 'flower_photos'

#Downloading the data.
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('flower_photos.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Flowers Dataset') as pbar:
        urlretrieve(
            'http://download.tensorflow.org/example_images/flower_photos.tgz',
            'flower_photos.tar.gz',
            pbar.hook)

if not isdir(dataset_folder_path):
    with tarfile.open('flower_photos.tar.gz') as tar:
        tar.extractall()
        tar.close()

data_dir = 'flower_photos/'
directory = './'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

file_paths = []
labels = []
for each in classes:
	class_path = data_dir + each
	files = os.listdir(class_path)

	for i, file in enumerate(files):
		path = os.path.join(directory+'/'+class_path, file)
		target = os.path.join(directory, class_path +'/'+each+'_'+ str(i)+'.jpg')
		os.rename(path, target)

for each in classes:
	class_path = data_dir + each
	files = os.listdir(class_path)
	
	files = os.listdir(class_path)
	for file in files:
		file_paths.append(os.path.join(directory, class_path +'/'+file))
		labels.append(each)

print(len(file_paths), len(labels))

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_idx, val_idx = next(ss.split(file_paths, labels))
half_val_len = int(len(val_idx)/2)
print(half_val_len, val_idx[half_val_len:])
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

print('Train',len(train_idx))
print('Test',len(test_idx))
print('Validation',len(val_idx))
train_y = []
for i in train_idx:
	shutil.copy(file_paths[i],'./data/train/'+labels[i])
	train_y.append(labels[i])

val_y = []
for j in val_idx:
	shutil.copy(file_paths[j],'./data/validation/'+labels[j])
	val_y.append(labels[j])

test_y = []
for k in test_idx:
	shutil.copy(file_paths[k],'./data/test/'+labels[k])
	test_y.append(labels[k])

pickle.dump([train_y, val_y, test_y], open('preprocess.p','wb'))
print('Data converted into train, test and validation')