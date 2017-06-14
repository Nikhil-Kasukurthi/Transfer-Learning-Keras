from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import os
import pickle
import numpy as np

#Loading the pre-processed data
train_y, val_y, test_y = pickle.load(open('preprocess.p','rb'))

img_width, img_height = 150, 150
top_model_weights_path = 'bottleneck_fc_model.h5'
batch_size = 16
epochs = 50

def save_bottleneck_features():
	'''
		VGG16 without including the top fully connected layers.
		The model is trained and the codes are saved as numpy array.
	'''
	model = applications.VGG16(weights = 'imagenet', include_top = False)

	datagen = ImageDataGenerator(
			rescale=1./255,
			rotation_range = 40,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			shear_range = 0.2,
			zoom_range = 0.2,
			fill_mode = 'nearest'
			)

	generator = datagen.flow_from_directory(
				'./data/train',
				target_size = (img_width, img_height),
				batch_size = batch_size,
				class_mode = None,
				shuffle = False
				)

	bottleneck_features_train = model.predict_generator(generator, len(train_y)//batch_size)
	np.save(open('bottleneck_features_train.npy','wb'), bottleneck_features_train)

	generator = datagen.flow_from_directory(
				'./data/validation',
				target_size = (img_width, img_height),
				batch_size = batch_size,
				class_mode = None,
				shuffle = False
				)
	bottleneck_features_val = model.predict_generator(generator, len(val_y)//batch_size)
	np.save(open('bottleneck_features_val.npy','wb'), bottleneck_features_val)
	
def train_top_model():
	'''
		Custom layers are attached on top of the pre-trained model.
	'''
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    val_data = np.load(open('bottleneck_features_val.npy','rb'))
    classes_dict = {'roses':0,
                    'dandelion':1,
                    'sunflowers':2,
                    'tulilps':3,
                    'daisy':4
                   }
    train_one_hot = np.zeros([len(train_y),5])
    for index, value in enumerate(train_y):
        train_one_hot[index][classes_dict.get(value)] = 1
    
    val_one_hot = np.zeros([len(val_y),5])
    for index, value in enumerate(val_y):
        val_one_hot[index][classes_dict.get(value)] = 1
    
    model = Sequential()
    model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_one_hot,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(val_data,val_one_hot))
    model.save_weights(top_model_weights_path)


save_bottleneck_features()
train_top_model()