import torch
 
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import cv2
import pickle
import h5py
import numpy as np
import time
# Load the configuration and model checkpoint
config = './Video_Swin_Transformer/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py'
checkpoint = './Video_Swin_Transformer/checkpoints/swin_small_patch244_window877_kinetics400_1k.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
model.cuda()
model.eval() 
#model.float()
# Load the checkpoint onto the GPU
checkpoint = load_checkpoint(model, checkpoint, map_location='cuda')
 
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)   # Add crop also 

def center_crop_frame(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx, :]

pkl_file_path = 'MSVD_test_list.pkl'          #######################################################   
# Read the contents of the .pkl file

c = 0
with open(pkl_file_path, 'rb') as file:
    video_filenames = pickle.load(file)

# Initialize an empty list to store video features
video_features = []

# Define the number of frames you want to sample uniformly
num_frames_to_sample = 8

  
h5 = h5py.File('MSVD_swin_S_8_test.hdf5', 'w')   # out file 
for video_filename in video_filenames:  
    vid = video_filename
    video_filename = video_filename + '.avi'                      #######################################################   

    
###########################################################################################################   
 
    
    cap = cv2.VideoCapture(f'/your path for dataset folder/{video_filename}')

###########################################################################################################
    # Get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 #   print('0000000000', num_frames)
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
        frame = cv2.resize(frame, (320, 240))
       
        frame = center_crop_frame(frame, 224, 224)
        frame = frame / 255.0
        frames.append(frame)
 
    
    frame_list = np.array(frames, dtype=np.float32)
    
     
    frame_list = (frame_list - np.array(mean)) / np.array(std)
    
    frame_list = torch.stack([torch.tensor(frame2).float() for frame2 in frame_list])
   
         
    frame_list = frame_list.permute(3, 0, 1, 2).unsqueeze(0).cuda()
  
    with torch.no_grad():
        start_time = time.time()
        features = model.extract_feat(frame_list) # -> torch.Size([1, 768, 8, 7, 7])
        features = features.mean(dim=[2,3,4])
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        if features is not None:
            h5[vid] = features.detach().cpu().numpy()
        
    

h5.close()
