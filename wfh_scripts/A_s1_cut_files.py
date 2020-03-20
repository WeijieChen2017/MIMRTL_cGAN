import os
import glob
import numpy as np
import nibabel as nib
import random


# create directory
name_dataset = "3dunet"

for folder_name in ["trainA", "trainB", "testA", "testB", "train", "test"]:
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

def SpotTheDifference_Generator(dataA, dataB, name_dataset, n_slice=3, name_tag="",
								num_sample=1000, remove_background=True, cube_size=64):
    # shape supposed to be 512*512*284 by default
    assert dataA.shape == dataB.shape, ("DataA should share the same shape with DataB.")
    path2save = "../pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/train/"
    x, y, z = dataA.shape

    # N, C, D, H, W
    output_cube = np.zeros((1, cube_size, cube_size, cube_size*2))
    pure_cube = np.zeros((cube_size, cube_size, cube_size))
    blur_cube = np.zeros((cube_size, cube_size, cube_size))

    for idx in range(num_sample):
        Bx = random.randint(0, x-1)
        By = random.randint(0, y-1)
        Bz = random.randint(0, z-1)
        Ex, Ey, Ez = Bx + cube_size, By + cube_size, Bz + cube_size         
        pure_cube = dataA[Bx:Ex, By:Ey, Bz:Ez]
        blur_cube = dataB[Bx:Ex, By:Ey, Bz:Ez]
        output_cube[0, :, :, :cube_size] = pure_cube
        output_cube[0, :, :, cube_size:] = blur_cube
        save_name = name_tag+"_Bx"+str(Bx)+"_By"+str(By)+"_Bz"+str(Bz)+".npy"
        np.save(path2save+save_name, output_cube)
        print(idx, path2save+save_name)



list_ori = glob.glob("../data/"+name_dataset+"/pure/*.nii")
list_ori.sort()
for path_ori in list_ori:
    print("TrainA:")
    filename_ori = os.path.basename(path_ori)[:]
    filename_ori = filename_ori[:filename_ori.find(".")]
    print(filename_ori)
    data_ori = maxmin_norm(nib.load(path_ori).get_fdata())
    
    list_sim = glob.glob("../data/"+name_dataset+"/blur/*"+filename_ori+"*.nii")
    list_sim.sort()
    
    for path_sim in list_sim:
        print("Pairs")
        filename_sim = os.path.basename(path_sim)[:]
        filename_sim = filename_sim[:filename_sim.find(".")]
        print("A:", filename_ori)
        print("B:", filename_sim)
                
        data_sim = maxmin_norm(nib.load(path_sim).get_fdata())
        SpotTheDifference_Generator(dataA=data_ori, dataB=data_sim,
                                    name_dataset=name_dataset, name_tag=filename_sim)
        
    print("------------------------------------------------------------------------")