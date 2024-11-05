from os.path import exists, join
from RandLANet import Network
from helper_ply import read_ply
from helper_tool import ConfigSCU as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
from helper_ply import write_ply
from os import makedirs
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import time, pickle, argparse, glob, os
import getpass


class Sensat:
    def __init__(self, test_area_idx):
        self.name = "Sensat"
        self.path = "./data"
        self.label_to_names = {
            0: "else",
            1: "road"
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        # self.val_split = "cambridge_block_20"
        self.val_split = "test"
        self.all_files = glob.glob(join(self.path, "original_ply", "*.ply"))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {"training": [], "validation": []}
        self.input_colors = {"training": [], "validation": []}
        self.input_labels = {"training": [], "validation": []}
        self.input_normals = {"training": [], "validation": []}
        self.input_names = {"training": [], "validation": []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, "input_{:.3f}".format(sub_grid_size))
        for i, file_path in enumerate(
            self.all_files
        ):  # all_files： Area1+Area23456共44+228个ply文件作为训练集
            t0 = time.time()
            cloud_name = file_path.split("/")[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = "validation"
            else:
                cloud_split = "training"

            # Name of the input files
            kd_tree_file = join(tree_path, "{:s}_KDTree.pkl".format(cloud_name))
            sub_ply_file = join(tree_path, "{:s}.ply".format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data["red"], data["green"], data["blue"])).T
            sub_labels = data["class"]
            sub_normals = np.vstack((data["normal_x"],data["normal_y"],data["normal_z"])).T

            # Read pkl with search tree
            with open(kd_tree_file, "rb") as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_normals[cloud_split] += [sub_normals]
            self.input_names[cloud_split] += [cloud_name]  # Area23456共228个ply文件作为训练集

            size = sub_colors.shape[0] * 4 * 7
            print(
                "{:s} {:.1f} MB loaded in {:.1f}s".format(
                    kd_tree_file.split("/")[-1], size * 1e-6, time.time() - t0
                )
            )

        print("\nPreparing reprojected indices for testing")

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split("/")[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, "{:s}_proj.pkl".format(cloud_name))
                with open(proj_file, "rb") as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print("{:s} done in {:.1f}s".format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == "training":
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == "validation":
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(
                    scale=cfg.noise_init / 10, size=center_point.shape
                )
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                # print("num_points = ")
                # print(len(points))
                # print("******************")
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(
                        pick_point, k=len(points)
                    )[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(
                        pick_point, k=cfg.num_points
                    )[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]
                queried_pc_normals = self.input_normals[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(
                    np.square((points[queried_idx] - pick_point).astype(np.float32)),
                    axis=1,
                )
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(
                    np.min(self.possibility[split][cloud_idx])
                )

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    (
                        queried_pc_xyz,
                        queried_pc_colors,
                        queried_idx,
                        queried_pc_labels,
                        queried_pc_normals
                    ) = DP.data_aug(
                        queried_pc_xyz,
                        queried_pc_colors,
                        queried_pc_labels,
                        queried_pc_normals,
                        queried_idx,
                        cfg.num_points,
                    )

                if True:
                    yield (
                        queried_pc_xyz.astype(np.float32),
                        queried_pc_colors.astype(np.float32),
                        queried_pc_labels,
                        queried_idx.astype(np.int32),
                        np.array([cloud_idx], dtype=np.int32),
                        queried_pc_normals.astype(np.float32),
                    )

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32,tf.float32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None], [None, 3])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(
            batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_normals
        ):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            input_normals = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(
                    DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32 #为每一个点计算邻域
                )
                sub_points = batch_xyz[
                    :, : tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], : #给所有点降采样
                ]
                pool_i = neighbour_idx[
                    :, : tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], : #给所有邻居点降采样
                ]
                
                sub_normals = batch_normals[
                    :, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], : #给所有法向量降采样
                ]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)

                input_normals.append(batch_normals)

                batch_xyz = sub_points #变为降采样后的点

                batch_normals = sub_normals

            input_list = input_points + input_neighbors + input_pools + input_up_samples + input_normals
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print("Initiating input pipelines")
        cfg.ignored_label_inds = [
            self.label_to_idx[ign_label] for ign_label in self.ignored_labels
        ]
        gen_function, gen_types, gen_shapes = self.get_batch_gen("training")
        gen_function_val, _, _ = self.get_batch_gen("validation")
        self.train_data = tf.data.Dataset.from_generator(
            gen_function, gen_types, gen_shapes
        )
        self.val_data = tf.data.Dataset.from_generator(
            gen_function_val, gen_types, gen_shapes
        )

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(
            self.batch_train_data.output_types, self.batch_train_data.output_shapes
        )
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="the number of GPUs to use [default: 0]"
    )
    parser.add_argument(
        "--test_area",
        type=int,
        default=1,
        help="Which area to use for test, option: 1-6 [default: 5]",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="options: train, test, vis"
    )
    parser.add_argument(
        "--model_path", type=str, default="None", help="pretrained model path"
    )
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = Sensat(test_area)
    dataset.init_input_pipeline()

    if Mode == "train":
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == "test":
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path is not "None":
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort(
                [
                    os.path.join("results", f)
                    for f in os.listdir("results")
                    if f.startswith("Log")
                ]
            )
            chosen_folder = logs[-1]
            # chosen_folder = os.path.join("results", "Log_2024-01-17_06-42-25")
            snap_path = join(chosen_folder, "snapshots")
            snap_steps = [
                int(f[:-5].split("-")[-1])
                for f in os.listdir(snap_path)
                if f[-5:] == ".meta"
            ]
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, "snap-{:d}".format(chosen_step))
        # tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    else:
        ##################
        # Visualize data # 用的train 的bach
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            i=0
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                print("len = ",len(dataset.input_names["validation"]))
                name = dataset.input_names["validation"][i] + ".ply"
                i=i+1
                save_path =  join("test_log/Log_2023_2_28", "val_preds")
                print(name)
                print("--------------------------------------okk")
                makedirs(save_path) if not exists(save_path) else None
                write_ply(
                                join(save_path, name),
                                [pc_xyz[0, :, :], labels[0, :]],
                                ["x","y","z", "label"],
                            )
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
