# Transfer learning using Keras
Using the pretrained model of VGG16, we build this model. 
For a more deatiled understanding check the blog on [S/R](https://www.skcript.com/svr/leveraging-on-transfer-learning-for-image-classification-using-keras/)

### Requirements
Python 3.5
Keras 2.0.4
Numpy 1.12.0

### Pre-processing the data
Run the data.sh bash file
```
bash data.sh
```
This command is used to create the folder necessary for training. 

```
python data.py
```
This command donwloads the data and performs the pre-processing.

The files are renamed in a chronlogical order.
Then they are segmented into *data/train*, *data/test*, *data/validation*
Then the labels are saved as pickle. 

### Training the model
The VGG16 model with pre-trained weights of imagenet is trained on the dataset
Check the [blog](https://www.skcript.com/svr/leveraging-on-transfer-learning-for-image-classification-using-keras/) for further understanding.
