from os.path import exists, join
from RandLANet import Network
from tester_SensatUrban import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigSensatUrban
from helper_tool import DataProcessing as DP
from helper_tool import Plot
from helper_ply import write_ply
from os import makedirs
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import time, pickle, argparse, glob, os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
import logging

cfg = ConfigSensatUrban()

class SensatUrban:
    def __init__(self, test_area_idx):
        self.name = 'SensatUrban'
        self.path = './data'
        # self.label_to_names = {0: 'ground',
        #                        1: 'vegetation',
        #                        2: 'building',
        #                        3: 'wall',
        #                        4: 'bridge',
        #                        5: 'parking',
        #                        6: 'rail',
        #                        7: 'traffic_road',
        #                        8: 'street_furniture',
        #                        9: 'car',
        #                        10: 'footpath',
        #                        11: 'bike',
        #                        12: 'water'
        # }
        self.label_to_names = {0: 'road',
                                1: 'else',
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        # self.val_split = 'Area_' + str(test_area_idx)
        self.val_split = "val"
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_xyz = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_xyz = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_normals = {'training': [], 'validation': []}
        self.input_curvatures = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_normals = np.vstack((data['normal_x'], data['normal_y'], data['normal_z'])).T
            sub_curvatures = data['curvatures']
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_xyz[cloud_split] += [sub_xyz]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_normals[cloud_split] += [sub_normals]
            self.input_curvatures[cloud_split] += [sub_curvatures]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 11
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels, xyz = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                self.val_xyz += [xyz]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
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
                # Get data source file name
                filename = self.input_names[split][cloud_idx]

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)
                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_normals = self.input_normals[split][cloud_idx][queried_idx]
                queried_pc_curvatures = self.input_curvatures[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_pc_normals, queried_pc_curvatures, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_normals, queried_pc_curvatures, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_normals.astype(np.float32),
                           queried_pc_curvatures.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None, 3], [None], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(batch_xyz, batch_colors, batch_normals, batch_curvatures, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_curvatures = tf.expand_dims(batch_curvatures, axis=-1)
            batch_features = tf.concat([batch_xyz, batch_colors], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            input_normals = []
            input_curvatures = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], Tout=tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                sub_normals = batch_normals[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                sub_curvatures = batch_curvatures[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]

                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], Tout=tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                input_normals.append(batch_normals)
                input_curvatures.append(batch_curvatures)

                batch_xyz = sub_points
                batch_normals = sub_normals
                batch_curvatures = sub_curvatures

            input_list = input_points + input_neighbors + input_pools + input_up_samples + input_normals + input_curvatures
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=1, help='Which area to use for test, option: 1-6 [default: 1]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode
    test_area = FLAGS.test_area

    if Mode == 'auto':
         # 定义超参数搜索空间
        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(1e-2)),  # 学习率
            'k_n': hp.choice('k_n', [12, 13, 14, 15, 16, 17, 18, 19, 20]),  # KNN 超参数
            'batch_size': hp.choice('batch_size', [4, 6, 8, 10]),  # batch size
            'noise_init': hp.uniform('noise_init', 3.0, 4.0),  # 噪声初始参数
            'num_layers': hp.choice('num_layers', [3, 4, 5, 6]),  # 网络层数

            # 你可以根据需要添加更多的超参数
            # k_n = 16  # KNN
            # num_layers = 5  # Number of layers
            # num_points = 40960  # Number of input points
            # num_classes = 2
            # sub_grid_size = 0.04  # preprocess_parameter
            # batch_size = 8  # batch_size during training
            # val_batch_size = 20  # batch_size during validation and test
            # train_steps = 900  # Number of steps per epochs
            # val_steps = 100  # Number of validation steps per epoch
            # sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
            # d_out = [16, 64, 128, 256, 512]  # feature dimension
            # noise_init = 3.5  # noise initial parameter
            # max_epoch = 50  # maximum epoch during training
            # learning_rate = 1e-2  # initial learning rate
            # lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
            # train_sum_dir = 'train_log'
            # saving = True
            # saving_path = None
        }

        # 目标函数，返回多个指标并进行加权组合
        def objective(params):
            # 更新配置
            cfg.saving = True
            cfg.learning_rate = params['learning_rate']
            cfg.k_n = params['k_n']
            cfg.batch_size = params['batch_size']
            # 根据 num_layers 调整 sub_sampling_ratio 和 d_out 的长度
            if cfg.num_layers == 3:
                cfg.sub_sampling_ratio = [4, 4, 2]
                cfg.d_out = [16, 64, 128,]
            elif cfg.num_layers == 4:
                cfg.sub_sampling_ratio = [4, 4, 4, 4, 2]
                cfg.d_out = [16, 64, 128, 256]
            elif cfg.num_layers == 5:
                cfg.sub_sampling_ratio = [4, 4, 4, 4, 2]
                cfg.d_out = [16, 64, 128, 256, 512]
            elif cfg.num_layers == 6:
                cfg.sub_sampling_ratio = [4, 4, 4, 4, 4, 2]
                cfg.d_out = [16, 64, 128, 256, 512, 1024]

            # 训练模型
            dataset = SensatUrban(test_area)
            dataset.init_input_pipeline()
            model = Network(dataset, cfg)
            model.train(dataset)
            # 测试模型并返回两个评判指标
            dataset = SensatUrban(test_area)
            tf.reset_default_graph()
            dataset.init_input_pipeline()

            cfg.saving = False
            model = Network(dataset, cfg)
            if FLAGS.model_path != 'None':
                chosen_snap = FLAGS.model_path
            else:
                chosen_snapshot = -1
                logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
                chosen_folder = logs[-1]
                snap_path = join(chosen_folder, 'snapshots')
                snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
                chosen_step = np.sort(snap_steps)[-1]
                chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
            tester = ModelTester(model, dataset, restore_snap=chosen_snap)
            # m_IoU = tester.test(model, dataset)
            # print("m_IoU:", m_IoU)
            m_IoU, m_Apls = tester.test(model, dataset)
            ObjFunc = m_IoU + 30 * m_Apls

            tf.reset_default_graph()
            return {'loss': -ObjFunc, 'status': STATUS_OK}
            #return {'loss': -m_Apls, 'status': STATUS_OK}
            # # 综合两个指标，比如加权平均
            # weight_accuracy = 0.7  # 权重，可以调整
            # weight_connectivity = 0.3  # 权重，可以调整
            # # 计算综合得分
            # combined_metric = weight_accuracy * accuracy + weight_connectivity * connectivity
            # 返回负的综合得分，越大越好，所以要返回负值以最小化目标

        # 创建一个 Trials 对象来存储优化过程中的所有信息
        # trials = Trials()
        # 执行贝叶斯优化
        cfg.max_epoch = 25
        best = fmin(
            fn=objective,  # 目标函数
            space=space,  # 超参数搜索空间
            algo=tpe.suggest,  # 贝叶斯优化算法
            max_evals=20  # 最大评估次数
            # trials=trials  # 用于存储结果
        )
        print("Best hyperparameters:", best)

    elif Mode == 'train':
        dataset = SensatUrban(test_area)
        dataset.init_input_pipeline()
        cfg.max_epoch = 50
        # cfg.k_n = 17
        # cfg.batch_size = 6
        # cfg.learning_rate = 0.00267
        # cfg.noise_init = 3.1
        # cfg.num_layers == 5

        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        dataset = SensatUrban(test_area)
        dataset.init_input_pipeline()
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path != 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    else:
        ##################
        # Visualize data #
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
                save_path =  join("train_log/2024_5_13", "val_preds")
                print(name)
                print("--------------------------------------ok")
                makedirs(save_path) if not exists(save_path) else None
                write_ply(
                                join(save_path, name),
                                [pc_xyz[0, :, :], labels[0, :]],
                                ["x","y","z", "label"],
                            )
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])

