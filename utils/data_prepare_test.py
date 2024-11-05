from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_ply import read_ply
from helper_tool import DataProcessing as DP

dataset_path = './data/sensat'

sub_grid_size = 0.4
sub_pc_folder = join(dirname(dataset_path), "sub_sampling")
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = ".ply"

def convert_pc2ply(anno_path, save_path):
    data = read_ply(anno_path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data["red"], data["green"], data["blue"])).T
    if "test" in anno_path:
        labels = np.zeros([len(xyz),1])
    else:
        labels = np.vstack((data["scalar_class"]))
    
    for i in range(len(labels)):
        label_int=int(float(labels[i]))
        if(np.equal(label_int,4) or np.equal(label_int,7) or np.equal(label_int,10)):
            labels[i]=0
        else:
            labels[i]=1

    xyz = xyz.astype(np.float32)
    colors = colors.astype(np.uint8)
    labels = labels.astype(np.uint8)
    # write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0

    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels],
        ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


if __name__ == '__main__':
    print("Start data_prepare:")
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    print("dataset_path:", dataset_path)
    for file_name in glob.glob(join(dataset_path, "*.ply")):
        elements = str(file_name).split('/')
        out_file_name = elements[-2] + '_' + elements[-1]
        print(out_file_name)
        convert_pc2ply(file_name, join(sub_pc_folder, out_file_name))

    print("Finish data_prepare!")
