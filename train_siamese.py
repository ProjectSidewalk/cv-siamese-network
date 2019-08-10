#%%
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
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


#%%
# Adapted from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
class SiameseDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        same_class = random.randint(0,1) 
        if same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) # Find a better way to do this
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) # This too
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


#%%
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return contrastive_loss


#%%
pbar_type = "bash"
epochs = 20
batch_size = 64

model_name = "skw.base_cnn.v1"
model_path = os.path.join("models",model_name+".ep_"+str(epochs))
if not os.path.exists(model_path):
    os.makedirs(model_path)

root_dir = "data"
train_dir = os.path.join(root_dir,"train")
test_dir = os.path.join(root_dir,"test")

im_train_dset = dset.ImageFolder(root=train_dir)
im_test_dset = dset.ImageFolder(root=test_dir)
transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()])
train_siamese_dset = SiameseDataset(imageFolderDataset=im_train_dset, transform=transformation)
val_siamese_dset = SiameseDataset(imageFolderDataset=im_test_dset, transform=transformation)
train_loader = DataLoader(train_siamese_dset,shuffle=True,num_workers=0,batch_size=batch_size)
val_loader = DataLoader(val_siamese_dset,shuffle=True,num_workers=0,batch_size=batch_size)

loss_function = ContrastiveLoss()

optimizer = optim.Adadelta(model.parameters())
print("CUDA availability:",torch.cuda.is_available())


#%%
start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

losses = []
batches = len(train_loader)
val_batches = len(val_loader)
best_loss = 0

model = SiameseNetwork().cuda()

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
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
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



