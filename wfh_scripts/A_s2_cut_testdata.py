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

    for idx_x in range((x-cube_size)//step_size+1):
        for idx_y in range((y-cube_size)//step_size+1):
            for idx_z in range((z-cube_size)//step_size+1):
                Bx1, By1, Bz1 = idx_x*step_size, idx_y*step_size, idx_z*step_size
                Ex1, Ey1, Ez1 = Bx1+cube_size, By1+cube_size, Bz1+cube_size         

                Bx0, By0, Bz0 = Bx1, By1, Bz1-1
                Ex0, Ey0, Ez0 = Ex1, Ey1, Ez1-1
                Bx2, By2, Bz2 = Bx1, By1, Bz1+1
                Ex2, Ey2, Ez2 = Ex1, Ey1, Ez1+1

                Ez0 = Ez0+1 if Bz0 <0 else Ez0
                Bz0 = 0 if Bz0 < 0 else Bz0
                Bz2 = Bz2-1 if Ez2 >z else Bz2
                Ez2 = z if Ez2 > z else Ez2

                output_cube[0, :, :, :] = dataA[Bx0:Ex0, By0:Ey0, Bz0:Ez0]
                output_cube[1, :, :, :] = dataA[Bx1:Ex1, By1:Ey1, Bz1:Ez1]
                output_cube[2, :, :, :] = dataA[Bx2:Ex2, By2:Ey2, Bz2:Ez2]

                cube_mean = np.mean(output_cube)
                save_name = path2save+"X"+str(Bx)+"Y"+str(By)+"Z"+str(Bz)+"_C"+str(cube_size)+"S"+str(step_size)+"_pure.npy"
                np.save(save_name, output_cube)
                print(idx_x, idx_y, idx_z, cube_mean)
     
    # extra patches for z-axis
    for idx_x in range((pure_data.shape[0]-cube_size)//step_size+1):
        for idx_y in range((pure_data.shape[1]-cube_size)//step_size+1):
            Bz1 = 220
            Bx1, By1 = idx_x*step_size, idx_y*step_size
            Ex1, Ey1, Ez1 = Bx1+cube_size, By1+cube_size, Bz1+cube_size         

            Bx0, By0, Bz0 = Bx1, By1, Bz1-1
            Ex0, Ey0, Ez0 = Ex1, Ey1, Ez1-1
            Bx2, By2, Bz2 = Bx1, By1, Bz1+1
            Ex2, Ey2, Ez2 = Ex1, Ey1, Ez1+1

            Ez0 = Ez0+1 if Bz0 <0 else Ez0
            Bz0 = 0 if Bz0 < 0 else Bz0
            Bz2 = Bz2-1 if Ez2 >z else Bz2
            Ez2 = z if Ez2 > z else Ez2
            
            output_cube[0, :, :, :] = dataA[Bx0:Ex0, By0:Ey0, Bz0:Ez0]
            output_cube[1, :, :, :] = dataA[Bx1:Ex1, By1:Ey1, Bz1:Ez1]
            output_cube[2, :, :, :] = dataA[Bx2:Ex2, By2:Ey2, Bz2:Ez2]

            cube_mean = np.mean(output_cube)
            save_name = path2save+"X"+str(Bx)+"Y"+str(By)+"Z"+str(Bz)+"_C"+str(cube_size)+"S"+str(step_size)+"_pure.npy"
            np.save(save_name, output_cube)
            print(idx_x, idx_y, cube_mean)

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