#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'SVM_Testing'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import pandas as pd
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm as tqdm_jupyter
from tqdm import tqdm as tqdm_bash
from PIL import Image
import os
import random
from skimage.feature import hog
import pickle

#%%
def rgb2gray(rgb):
    # From: https://stackoverflow.com/q/12201577/6454085
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
#%%
train_dir = "../data/train"
test_dir = "../data/test"

filename = 'skewed.penalized.svm'

balanced = False

train_loader = list(os.walk(train_dir))
test_loader = list(os.walk(test_dir))

pbar_type = "bash"
#%%
train_images = []
train_labels = []
train_minSize = -1

for (r,d,f) in train_loader:
    if f:
        if len(f) < train_minSize or train_minSize == -1|:
            train_minSize = len(f)
print("loading training images")
for (r,d,f) in train_loader:
    if f:
        dirName = r.split("\\")[1]
        print("loading folder {}".format(dirName))

        random.shuffle(f)
        if balanced:
            f = f[:train_minSize]
        train_labels += [dirName]*len(f)

        if pbar_type == "bash":
            folderProgress = tqdm_bash(f, total = len(f), unit=" image(s)", position=0, leave=True)
        if pbar_type == "jupyter":
            folderProgress = tqdm_jupyter(f, total = len(f), unit=" image(s)")
        
        for ff in folderProgress:
            im = Image.open(os.path.join(r,ff))
            im = im.resize((64,64), Image.ANTIALIAS)
            train_images.append(np.array(im))
#%%
train_features = []

if pbar_type == "bash":
    imageProgress = tqdm_bash(train_images, total = len(train_images), unit=" image(s)", position=0, leave=True)
if pbar_type == "jupyter":
    imageProgress = tqdm_jupyter(train_images, total = len(train_images), unit=" image(s)")
for img in imageProgress:
    gray_img = rgb2gray(img)
    hog_features = hog(
        gray_img, pixels_per_cell=(12, 12),
        cells_per_block=(2,2),
        orientations=8,
        block_norm='L2-Hys')
    train_features.append(hog_features)
#%%
test_images = []
test_labels = []
test_minSize = -1

for (r,d,f) in test_loader:
    if f:
        if len(f) < test_minSize or test_minSize == -1:
            test_minSize = len(f)
print("loading testing images")
for (r,d,f) in test_loader:
    if f:
        dirName = r.split("\\")[1]
        print("loading folder {}".format(dirName))

        random.shuffle(f)
        if balanced:
            f = f[:test_minSize]
        test_labels += [dirName]*len(f)

        if pbar_type == "bash":
            folderProgress = tqdm_bash(f, total = len(f), unit=" image(s)", position=0, leave=True)
        if pbar_type == "jupyter":
            folderProgress = tqdm_jupyter(f, total = len(f), unit=" image(s)")
        
        for ff in folderProgress:
            im = Image.open(os.path.join(r,ff))
            im = im.resize((64,64), Image.ANTIALIAS)
            test_images.append(np.array(im))

#%%
test_features = []

if pbar_type == "bash":
    imageProgress = tqdm_bash(test_images, total = len(test_images), unit=" image(s)", position=0, leave=True)
if pbar_type == "jupyter":
    imageProgress = tqdm_jupyter(test_images, total = len(test_images), unit=" image(s)")
for img in imageProgress:
    gray_img = rgb2gray(img)
    hog_features = hog(
        gray_img, pixels_per_cell=(12, 12),
        cells_per_block=(2,2),
        orientations=8,
        block_norm='L2-Hys')
    test_features.append(hog_features)

#%%
classifier = svm.SVC(gamma=0.001,class_weight="balanced")
classifier.fit(train_features,train_labels)

#%%
predicted = classifier.predict(test_features)
print(metrics.classification_report(predicted,test_labels))

#%%
pickle.dump(classifier, open(filename, 'wb'))
#%%
pickle.dump((test_images,test_features,test_labels), open('test_data.pkl','wb'))
pickle.dump((train_images,train_features,train_labels), open('train_data.pkl','wb'))
#%%
