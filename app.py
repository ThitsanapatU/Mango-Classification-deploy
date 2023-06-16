import cv2
import numpy as np
import streamlit as st
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from torchvision.io import read_image
from PIL import Image


st.set_page_config(layout="centered")
st.title("Thai Mango Image Classification")

uploaded_file = st.file_uploader("Choose a image file", type="jpg")


#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=3):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (32,3,200,200)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (32,12,200,200)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (32,12,200,200)
        self.relu1=nn.ReLU()
        #Shape= (32,12,200,200)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (32,12,100,100)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (32,20,100,100)
        self.relu2=nn.ReLU()
        #Shape= (32,20,100,100)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (32,32,100,100)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (32,32,100,100)
        self.relu3=nn.ReLU()
        #Shape= (32,32,100,100)
        
        
        self.fc=nn.Linear(in_features=100 * 100 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (32,32,100,100)
            
        output=output.view(-1,32*100*100)
            
            
        output=self.fc(output)
            
        return output
checkpoint=torch.load('best_checkpoint.model')
model=ConvNet(num_classes=4)
model.load_state_dict(checkpoint)
model.eval()

#Transforms
transformer=transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])
#prediction function
def prediction(img_path,transformer):
    
    image=Image.open(img_path)
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    
    output=model(input)
    
    index=output.data.numpy().argmax()
    
    pred=classes[index]
    
    return pred

classes = ['R2E2', 'green', 'mahachanok', 'namdokmai']
model.food_class = classes

#Transforms
transformer=transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

if uploaded_file is not None:
    temp_file_path = "temp.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    img = Image.open(temp_file_path)

    # Display the input image
    st.image(img, caption='Mango Image', use_column_width=True)

    scale = transformer(img)
    torch_images = scale.unsqueeze(0)

    with torch.no_grad():
        outputs = model(torch_images)

        _, predict = torch.max(outputs, 1)
        pred_id = predict.item()
        
        st.write('Predicted Type : ', model.food_class[pred_id])

    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")