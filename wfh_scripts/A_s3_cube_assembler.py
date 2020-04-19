import os
import glob
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def get_cube_idx(path_cube, edge_length):
    idx_x = path_cube.rfind('X')
    idx_y = path_cube.rfind('Y')
    idx_z = path_cube.rfind('Z')
    idx_zz = path_cube[idx_z:].find('_')
    x = int(path_cube[idx_x+1:idx_y])
    y = int(path_cube[idx_y+1:idx_z])
    z = int(path_cube[idx_z+1:idx_z+idx_zz])
    return x,y,z

name_dataset = "z_r100e20"
test_folder = "test"
date_tag = name_dataset
edge_length = 64
count_cube = np.ones((edge_length, edge_length, edge_length))

list_ori = glob.glob("../data/3dunet/test/*.nii")
list_ori.sort()
for path_ori in list_ori:
    print(path_ori)
    nii_name = os.path.basename(path_ori)[:-4]
    nii_file = nib.load(path_ori)
    nii_data = nii_file.get_fdata()
    
    list_cubes = glob.glob("../pytorch-CycleGAN-and-pix2pix/results/"+name_dataset+"/test_latest/images/*_fake*.npy")
    list_cubes.sort()
    num_cubes = len(list_cubes)
    fake_value = np.zeros((nii_data.shape[0], nii_data.shape[1], nii_data.shape[2]))
    fake_count = np.zeros((nii_data.shape[0], nii_data.shape[1], nii_data.shape[2]))
    for idx, path_cube in enumerate(list_cubes):
        print(idx, "/", num_cubes, path_cube)
        data_cube = np.load(path_cube)    
        start_x, start_y, start_z = get_cube_idx(path_cube, edge_length)
        print(start_x, start_y, start_z, np.mean(data_cube))
        fake_value[start_x:start_x+edge_length, start_y:start_y+edge_length, start_z:start_z+edge_length] += data_cube[1, :, :, :]
        fake_count[start_x:start_x+edge_length, start_y:start_y+edge_length, start_z:start_z+edge_length] += count_cube
        
    # assert (not 0 in fake_count), ("Each pixel should be generated at least once.")
    
    pred_fake = np.divide(fake_value, fake_count)
    factor_f = np.sum(nii_file.get_data())/np.sum(pred_fake)
    file_fake = nib.Nifti1Image(pred_fake*factor_f, nii_file.affine, nii_file.header)
    nib.save(file_fake, "../"+nii_name+"_fake_value_"+date_tag+".nii")
    file_fake = nib.Nifti1Image(fake_count, nii_file.affine, nii_file.header)
    nib.save(file_fake, "../"+nii_name+"_fake_count_"+date_tag+".nii")