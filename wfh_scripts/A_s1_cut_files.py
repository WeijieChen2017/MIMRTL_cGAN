import os
import glob
import numpy as np
import nibabel as nib

name_dataset = "3dunet"

for folder_name in ["trainA", "trainB", "testA", "testB"]:
    path = "../pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/"
    if not os.path.exists(path):
        os.makedirs(path)
        
blur_path = "../data/"+name_dataset+"/blur/"
if not os.path.exists(blur_path):
    os.makedirs(blur_path)
    
pure_path = "../data/"+name_dataset+"/pure/"
if not os.path.exists(pure_path):
    os.makedirs(pure_path)