import os
import glob
import numpy as np
import nibabel as nib
import random


# create directory
name_dataset = "3dpet_norm"

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

def GenerateLegalCoordinates(x,y,z, cube_size):
    Bx = random.randint(0, x-cube_size-1)
    Ex = Bx + cube_size
    By = random.randint(0, y-cube_size-1)
    Ey = By + cube_size
    Bz = random.randint(1, z-cube_size-2)
    Ez = Bz + cube_size   
    return Bx, Ex, By, Ey, Bz, Ez


def Testdataset_Generator(dataA, name_dataset, n_slice=3, name_tag="",
                          remove_background=False, cube_size=64, step_size=64):
    # shape supposed to be 512*512*284 by default
    path2save = "../pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/test/"
    x, y, z = dataA.shape

    # N, C, D, H, W
    output_cube = np.zeros((3, cube_size, cube_size, cube_size))
    pure_cube = np.zeros((cube_size, cube_size, cube_size))

    for idx_x in range((x-cube_size)//step_size+1):
        for idx_y in range((y-cube_size)//step_size+1):
            for idx_z in range((z-cube_size)//step_size+1):
                Bx, By, Bz = idx_x*step_size, idx_y*step_size, idx_z*step_size
                Ex, Ey, Ez = Bx + cube_size, By + cube_size, Bz + cube_size         
                pure_cube = pure_data[Bx:Ex, By:Ey, Bz:Ez]
                cube_mean = np.mean(pure_cube)
                pure_name = pure_save_path+"X"+str(Bx)+"Y"+str(By)+"Z"+str(Bz)+"_C"+str(cube_size)+"S"+str(step_size)+"_pure.npy"
                np.save(pure_name, pure_cube)
                print(idx_x, idx_y, idx_z, cube_mean)
     
    # extra patches for z-axis
    for idx_x in range((x-cube_size)//step_size+1):
        for idx_y in range((y-cube_size)//step_size+1):
            Bz = 220
            Bx, By = idx_x*step_size, idx_y*step_size
            Ex, Ey, Ez = Bx + cube_size, By + cube_size, Bz + cube_size         
            pure_cube = pure_data[Bx:Ex, By:Ey, Bz:Ez]
            cube_mean = np.mean(pure_cube)
            pure_name = pure_save_path+"X"+str(Bx)+"Y"+str(By)+"Z"+str(Bz)+"_C"+str(cube_size)+"S"+str(step_size)+"_pure.npy"
            np.save(pure_name, pure_cube)
            print(idx_x, idx_y, cube_mean)

        output_cube[0, :, :, :cube_size] = dataA[Bx:Ex, By:Ey, Bz-1:Ez-1]
        output_cube[1, :, :, :cube_size] = pure_cube
        output_cube[2, :, :, :cube_size] = dataA[Bx:Ex, By:Ey, Bz+1:Ez+1]

        save_name = name_tag+"_Bx"+str(Bx)+"_By"+str(By)+"_Bz"+str(Bz)+".npy"
        np.save(path2save+save_name, output_cube)
        print(idx, path2save+save_name)

list_ori = glob.glob("../data/"+name_dataset+"/pure/*.nii")
list_ori.sort()
for path_ori in list_ori:
    print("Test:")
    filename_ori = os.path.basename(path_ori)[:]
    filename_ori = filename_ori[:filename_ori.find(".")]
    print(filename_ori)
    data_ori = maxmin_norm(nib.load(path_ori).get_fdata())

    Testdataset_Generator(dataA=data_ori, name_dataset=name_dataset, name_tag=filename_ori)
        
    print("------------------------------------------------------------------------")