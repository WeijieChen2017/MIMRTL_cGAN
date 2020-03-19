import os
import glob
import numpy as np
import nibabel as nib


# create directory
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


# start load and cut
def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

def SpotTheDifference_Generator(dataA, dataB, name_dataset, n_slice=1, name_tag="", resize_f=1):
    # shape supposed to be 512*512*284 by default
    assert dataA.shape == dataB.shape, ("DataA should share the same shape with DataB.")
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/train/"
    h, w, c = dataA.shape
    h = h*resize_f
    w = w*resize_f
    img = np.zeros((n_slice, h, w*2))
        
list_ori = glob.glob("./data/"+name_dataset+"/pure/*.nii")
list_ori.sort()
for path_ori in list_ori:
    print("TrainA:")
    filename_ori = os.path.basename(path_ori)[:]
    filename_ori = filename_ori[:filename_ori.find(".")]
    print(filename_ori)
    data_ori = maxmin_norm(nib.load(path_ori).get_fdata())
    
    list_sim = glob.glob("./data/"+name_dataset+"/blur/*"+filename_ori+"*.nii")
    list_sim.sort()
    
    for path_sim in list_sim:
        print("Pairs")
        filename_sim = os.path.basename(path_sim)[:]
        filename_sim = filename_sim[:filename_sim.find(".")]
        print("A:", filename_ori)
        print("B:", filename_sim)
                
        data_sim = maxmin_norm(nib.load(path_sim).get_fdata())
        SpotTheDifference_Generator(dataA=data_ori, dataB=data_sim,
                                    name_dataset=name_dataset, n_slice=n_slice, 
                                    name_tag=filename_sim, resize_f=2)
        
    print("------------------------------------------------------------------------")