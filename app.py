import streamlit as st
import pandas as pd
import numpy as np
from torch import nn
from torchvision import models
from Predict import validation_dataset
from Predict import predict

st.set_page_config(page_title = "Deepfake Detection")

video = st.file_uploader(label = "Upload a video file", type = "mp4", accept_mnultiple_files = False)

path_to_model = 'C:/Users/anand/Downloads/MINOR PROJECT/Code/Trained model/ResNet50_20.pt'

class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 512, lstm_layers=1 , hidden_dim = 512, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnet34(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(512,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,512)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))
    
    
im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
model = Model(2).cuda()
model.load_state_dict(torch.load(path_to_model))
video_data = validation_dataset(video, sequence_length = 20, transformer = train_transforms)
prediction = predict(model, video_data)
if prediction == 1:
    print("Real")
else:
    print("Fake")
