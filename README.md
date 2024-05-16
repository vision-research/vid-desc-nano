
# Visually Impaired Assistive Tool 
This project aims to assist visually impaired individuals by generating audio descriptions of their environment using object detection and video description techniques. It has been tested and deployed on the NVIDIA Jetson Nano 4GB.


# Usage
Our project is implemented with PyTorch framework

### Environment
- Python = 3.6.9
- PyTorch = 1.8.
- OpenCV = 4.5.3 with GPU support

### Jetson Nan setup
For getting a Pre-installed frameworks with GPU support for the Jetson Nano board, download the Jetson Nano image file provided in the following repo: https://github.com/Qengineering/Jetson-Nano-image.git


### Software setup

1- Clone this repo https://github.com/vision-research/vid-desc-nano.git

2- In Yolov7 folder, Clone the yolov7 repo: https://github.com/WongKinYiu/yolov7.git

3- Replace the detect.py file with modified vesion in this repo to return an audio output rather than text

4- In Video_Swin_Transformer folder, Clone the following repo: https://github.com/SwinTransformer/Video-Swin-Transformer.git

5- Run this file for obtaining audio description:
```bash
  python Inference.py
```

### For downloading MSVD videos use link in:
https://www.cs.utexas.edu/users/ml/clamp/videoDescription/

the texts file are located in data folder in this repo



### For downloading MS COCO dataset:
https://cocodataset.org/#download


### For video captioning training run   

```bash
  python training_vid_cap.py
```
or use the trained model from this link: https://drive.google.com/drive/folders/1fm5rXDiMiPJhwdxjwihKtCiZlH1uBe0m?usp=sharing
then put the file in checkpoints folder  
### For feature extraction run   

```bash
  python Swin_feat_extraction.py
  python EffeB3_feat_extraction.py
```
or use extracted features from this link: https://drive.google.com/drive/folders/1IGZPW_e8MRQLG8qHlLiTf22sOvo8ssJd?usp=sharing
then put them in feature folder inside data/MSVD folder

# Results
Video description
![Res2](https://github.com/vision-research/vid-desc-nano/assets/169878400/cc66dd0f-9525-40d1-abab-deef088a6607)

Object detection using: a) yolov7-tiny  b) yolov7-W6
![obj](https://github.com/vision-research/vid-desc-nano/assets/169878400/0a2dc205-a907-42dc-a84e-b46d17bff14e)

    
## Acknowledgements


- https://github.com/Qengineering/Jetson-Nano-image
- https://github.com/SwinTransformer/Video-Swin-Transformer
- https://github.com/haofanwang/video-swin-transformer-pytorch
- https://github.com/WongKinYiu/yolov7?tab=readme-ov-file

Some code in this project is based on the following repositories:
- https://github.com/hobincar/RecNet
- https://github.com/v-iashin/BMT
