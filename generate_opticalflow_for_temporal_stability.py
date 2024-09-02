import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
import warnings

# Suppress specific UserWarning related to antialias
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional", lineno=1603)

# Load the RAFT model with pre-trained weights
weights = Raft_Large_Weights.DEFAULT
raft_model = raft_large(weights=weights)
raft_model = raft_model.cuda().eval()

# Define the transformation to preprocess the input images
transform = T.Compose([
    T.ToTensor(),
    T.Resize((512, 512)),  # Ensure image size is divisible by 8
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization to [-1, 1] is part of the weights transforms
])

def load_image(imfile):
    img = Image.open(imfile).convert('RGB')
    return transform(img).unsqueeze(0).cuda()

def generate_optical_flow(img1, img2):
    with torch.no_grad():
        # RAFT returns a list with the first element being the final upsampled flow
        list_of_flows = raft_model(img1, img2)
        flow_up = list_of_flows[-1]  # Get the final predicted flow
    return flow_up.permute(0, 2, 3, 1).cpu().numpy()[0]

# Define the root paths for the input images and the output optical flow
root_path = '/home/zhxing/Datasets/ViSha/test/labels'  # prediction root dir
flow_root_path = '/home/zhxing/Datasets/ViSha/test/labels_flow_numpy/'  # optical flow output dir
all_name = os.listdir(root_path)  # names of all videos

for name in tqdm(all_name):
    all_img = os.listdir(os.path.join(root_path, name))
    all_img.sort()
    os.makedirs(os.path.join(flow_root_path, name), exist_ok=True)
    
    for idex in range(len(all_img) - 1):
        img_name1 = all_img[idex]
        img_name2 = all_img[idex + 1]
        img_path1 = os.path.join(root_path, name, img_name1)
        img_path2 = os.path.join(root_path, name, img_name2)
        
        img1 = load_image(img_path1)
        img2 = load_image(img_path2)
        
        flow = generate_optical_flow(img1, img2)
        
        flow_name = f"{img_name1.split('.')[0]}_{img_name2.split('.')[0]}.npy"
        flow_path = os.path.join(flow_root_path, name, flow_name)
        np.save(flow_path, flow)

print("Optical flow generation complete.")
