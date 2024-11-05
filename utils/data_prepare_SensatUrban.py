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

sub_grid_size = 0.040
original_pc_folder = join(dirname(dataset_path), "original_ply")
sub_pc_folder = join(dirname(dataset_path), "input_{:.3f}".format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = ".ply"

def convert_pc2ply(anno_path, save_path):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    # data_list = []
    # for f in glob.glob(join(anno_path, '*.txt')):
    #     class_name = os.path.basename(f).split('_')[0]
    #     if class_name not in gt_class:  # note: in some room there is 'staris' class..
    #         class_name = 'clutter'
    #     pc = pd.read_csv(f, header=None, delim_whitespace=True).values
    #     labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
    #     data_list.append(np.concatenate([pc, labels], 1))  # Nx7
    # pc_label = np.concatenate(data_list, 0)
    # xyz_min = np.amin(pc_label, axis=0)[0:3]
    # pc_label[:, 0:3] -= xyz_min
    # xyz = pc_label[:, :3].astype(np.float32)
    # colors = pc_label[:, 3:6].astype(np.uint8)
    # labels = pc_label[:, 6].astype(np.uint8)

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

    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    # 计算并归一化特征
    sub_normals, curvatures = DP.calculate_pc_features(sub_xyz)
    # sub_normals, curvatures, edge_strengths, local_densities, sparsities, normal_variances = DP.calculate_pc_features(sub_xyz)
    # 归一化法向量
    sub_normals = (sub_normals + 1.0) / 2.0  # 将法向量的范围从 [-1, 1] 调整到 [0, 1]
    # 曲率标准化（可选择归一化或标准化）
    curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-5)  # [0, 1]归一化
    # # 边缘强度归一化
    # edge_strengths = (edge_strengths - np.min(edge_strengths)) / (np.max(edge_strengths) - np.min(edge_strengths) + 1e-5)
    # # 局部密度标准化
    # local_densities = (local_densities - np.min(local_densities)) / (np.max(local_densities) - np.min(local_densities) + 1e-5)
    # # 稀疏度归一化
    # sparsities = (sparsities - np.min(sparsities)) / (np.max(sparsities) - np.min(sparsities) + 1e-5)
    # # 法向量方差归一化
    # normal_variances = (normal_variances - np.min(normal_variances)) / (np.max(normal_variances) - np.min(normal_variances) + 1e-5)
    
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels, sub_normals, curvatures],
        ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'normal_x', 'normal_y', 'normal_z', 'curvatures'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels, xyz], f)


if __name__ == '__main__':
    print("Start data_prepare:")
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    print("dataset_path:", dataset_path)
    for file_name in glob.glob(join(dataset_path, "*.ply")):
        elements = str(file_name).split('/')
        out_file_name = elements[-2] + '_' + elements[-1]
        print(out_file_name)
        convert_pc2ply(file_name, join(original_pc_folder, out_file_name))

    print("Finish data_prepare!")
