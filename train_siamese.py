#%%
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, BasicBlock
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.autonotebook import tqdm as tqdm_jupyter
from tqdm import tqdm as tqdm_bash
import inspect
import os
import time
from siamese_tools import *
#%%
pbar_type = "bash"
epochs = 20
batch_size = 64

multi_gpu = False

model_name = "skw.base_resnet.v1"
model_path = os.path.join("models",model_name+".ep_"+str(epochs))
if not os.path.exists(model_path):
    os.makedirs(model_path)

root_dir = "data"
train_dir = os.path.join(root_dir,"train")
test_dir = os.path.join(root_dir,"test")

im_train_dset = dset.ImageFolder(root=train_dir)
im_test_dset = dset.ImageFolder(root=test_dir)

## Basic CNN Transformations
#transformation = transforms.Compose([transforms.Resize((100,100)),
#                                     transforms.ToTensor()])

## ResNet Transformations
transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

train_siamese_dset = SiameseDataset(imageFolderDataset=im_train_dset, transform=transformation)
val_siamese_dset = SiameseDataset(imageFolderDataset=im_test_dset, transform=transformation)
train_loader = DataLoader(train_siamese_dset,shuffle=True,num_workers=0,batch_size=batch_size)
val_loader = DataLoader(val_siamese_dset,shuffle=True,num_workers=0,batch_size=batch_size)

loss_function = ContrastiveLoss()

print("CUDA availability:",torch.cuda.is_available())


#%%
start_ts = time.time()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

losses = []
batches = len(train_loader)
val_batches = len(val_loader)
best_loss = 0

if torch.cuda.is_available():
    print("Emptying CUDA cache of {} cached bytes and {} allocated bytes".format(torch.cuda.memory_allocated(),torch.cuda.memory_cached()))
    torch.cuda.empty_cache()

#model = SiameseNetwork().to(device) ## Basic CNN Network
model = SiameseResNet().to(device) ## ResNet Network

if torch.cuda.device_count() > 1 and multi_gpu:
    print("Using", torch.cuda.device_count(), "GPUs for training")
    model = nn.DataParallel(model)

optimizer = optim.Adadelta(model.parameters())

for epoch in range(epochs):
    total_loss = 0
    
    progress = None
    
    if pbar_type == "bash":
        progress = tqdm_bash(enumerate(train_loader,0), total=batches, unit=" batches", desc="Loss: ", position=0, leave=True)
    if pbar_type == "jupyter":
        progress = tqdm_jupyter(enumerate(train_loader,0), total=batches, unit=" batches", desc="Loss: ")
    
    model.train()
    
    for i,data in progress:
        img0, img1 , label = data
        img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
        optimizer.zero_grad()
        output1, output2 = model(img0,img1)
        loss = loss_function(output1,output2,label)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss

        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    curr_loss = total_loss/batches
    print(f"Epoch {epoch+1}/{epochs}, training loss: {curr_loss}")
          
    torch.save(model.state_dict(),os.path.join(model_path,"epoch_"+str(epoch)+".pt"))
    if (curr_loss < best_loss) or (best_loss == 0):
        best_loss = curr_loss
        print("Saving new best model at epoch {}".format(epoch))
        torch.save(model.state_dict(),os.path.join(model_path,"best.pt"))
        
print(f"Training time: {time.time()-start_ts}s")
#%%



