import torch
import torch.nn as nn
import cv2
import pickle
import h5py
import numpy as np
from timm import create_model
import time


''' ####################

EfficientNet B0: 224x224 pixels  
EfficientNet B1: 240x240 pixels # out 1280
EfficientNet B2: 260x260 pixels # out  1408
EfficientNet B3: 300x300 pixels   # out 1536
 
''' #####################################


inp_size = 300
 
# Load pre-trained EfficientNet-B4 model
model = create_model('efficientnet_b3', pretrained='imagenet') #############################

# Remove the final classification layer
model = nn.Sequential(*list(model.children())[:-1])
model.cuda()
model.eval() 
 
 
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)   # Add crop also 

def center_crop_frame(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx, :]

pkl_file_path = 'MSVD_valid_list.pkl'          
# Read the contents of the .pkl file

c = 0
with open(pkl_file_path, 'rb') as file:
    video_filenames = pickle.load(file)

# Initialize an empty list to store video features
video_features = []

 
h5 = h5py.File('MSVD_B3_valid_8.hdf5', 'w')   # out file       #######################################################   

for video_filename in video_filenames:  
    vid = video_filename
    video_filename = video_filename + '.avi'                      #######################################################   

    
###########################################################################################################   
 
     
    cap = cv2.VideoCapture(f'/your path for dataset folder/{video_filename}')

###########################################################################################################
    # Get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    c += 1
    print(c)

     
 
###########################################################################################################    
    sampled_idxs = np.linspace(0, num_frames - 1, 8, dtype=int)   
    
 

###########################################################################################################    
    frames = []

    # Read frames from the video uniformly  for i in range(0, num_frames, step):
    for idx in sampled_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        
        if not ret:
            break

      
        frame = frame[:, :, ::-1]     
        frame = cv2.resize(frame, (460, 400))
       
        frame = center_crop_frame(frame, inp_size, inp_size)
        frame = frame / 255.0
        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
 
    
    frame_list = np.array(frames, dtype=np.float32)
   
     
    frame_list = torch.stack([torch.tensor(frame2).float() for frame2 in frame_list]).cuda()
    
   
    with torch.no_grad():
        
        start_time = time.time()
        features = model(frame_list)
        end_time = time.time()
        
        inference_time = end_time - start_time
    
        if features is not None:
            h5[vid] = features.detach().cpu().numpy()
        
    

h5.close()
