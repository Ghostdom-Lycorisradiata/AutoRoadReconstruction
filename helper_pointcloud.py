import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import sys
from PIL import Image

# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:

        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):
    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):

    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

            # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

        # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def read_image(path):
    img = plt.imread(path)
    return img

def write_image(image, path, name="result"):
    image = image.astype(np.uint8)
    
    # 确保目录存在，如果不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)

    print("Start save:")
    # 使用Pillow保存图像
    try:
        # 创建一个Image对象
        img = Image.fromarray(image)
        # 构建完整的文件路径，修改为.tif格式
        full_path = os.path.join(path, name + ".tif")
        # 保存图像为 TIFF 格式
        img.save(full_path, format='TIFF')
        print("Image saved successfully at {}".format(full_path))
        return full_path  # 返回保存的图像的完整路径
    except Exception as e:
        print("Failed to save image with Pillow:", e)
        return None  # 发生错误时返回 None


def whole_image_to_pointcloud_label(points, image, span_per_pixel, min_xy):
    start = time.time()

    points[:, :2] = points[:, :2] - min_xy

    x_pixel = points[:, 0] // span_per_pixel[0]  # 把每个点分配到一个像素（x_pixel保持points中点的个数和顺序，但存的值是每个点对应的x方向像素）
    y_pixel = points[:, 1] // span_per_pixel[1]

    resolution = [image.shape[0], image.shape[1]]
    x_pixel = np.where(x_pixel == resolution[0], resolution[0] - 1, x_pixel).astype(np.int32)  # 图像边界上的像素值-1
    y_pixel = np.where(y_pixel == resolution[1], resolution[1] - 1, y_pixel).astype(np.int32)

    labels = np.zeros(len(x_pixel))
    for i in range(len(labels)):
        if image[x_pixel[i]][y_pixel[i]][0] > 0.0:  # 如果[x, y]所在的像素为1.0，则label = 1
            labels[i] = 1

    print("Time Passed of whole_image_to_pointcloud_label: " + str((time.time() - start)/60) + " min")

    return labels


def image_to_pointcloud_label(points, image):
    start = time.time()

    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    resolution = [image.shape[0], image.shape[1]]
    span_per_pixel = (max_xy - min_xy) / resolution  # 计算每个像素的跨度

    points[:, :2] = points[:, :2] - min_xy

    x_pixel = points[:, 0] // span_per_pixel[0]  # 把每个点分配到一个像素（x_pixel保持points中点的个数和顺序，但存的值是每个点对应的x方向像素）
    y_pixel = points[:, 1] // span_per_pixel[1]

    x_pixel = np.where(x_pixel == resolution[0], resolution[0] - 1, x_pixel).astype(np.uint8)  # 图像边界上的像素值-1
    y_pixel = np.where(y_pixel == resolution[1], resolution[1] - 1, y_pixel).astype(np.uint8)

    labels = np.zeros(len(x_pixel))
    for i in range(len(labels)):
        if image[x_pixel[i]][y_pixel[i]][0] > 0.0:  # 如果[x, y]所在的像素为1.0，则label = 1
            labels[i] = 1

    print("Time Passed of image_to_pointcloud_label: " + str((time.time() - start)/60) + " min")

    return labels


def points_to_image(points, resolution):

    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    span_per_pixel = (max_xy - min_xy) / resolution  # 计算每个像素的跨度
    # print("min:", min_xy)
    # print("max:", max_xy)
    # print("span_per_pixel:", span_per_pixel)

    points[:, :2] = points[:, :2] - min_xy
    x_pixel = points[:, 0] // span_per_pixel[0]  # 把每个点分配到一个像素（x_pixel保持points中点的个数和顺序，但存的值是每个点对应的像素）
    y_pixel = points[:, 1] // span_per_pixel[1]
    x_pixel = np.where(x_pixel == resolution[0], resolution[0]-1, x_pixel)  # 图像边界上的像素值-1
    y_pixel = np.where(y_pixel == resolution[1], resolution[1]-1, y_pixel)
    points_grid = pd.DataFrame({"x_pixel": x_pixel, "y_pixel": y_pixel, "Z": points[:, 2]})
    points_grid = points_grid.groupby(["x_pixel", "y_pixel"]).idxmin().reset_index()  # 搜索最小的z的索引
    index = np.array(points_grid).astype(np.int32)

    labels = (points[index[:, 2], 3:4]).astype(np.uint8)  # 选择是label
    # print("Index shape:", index.shape) 
    # print("Labels shape:", labels.shape)
    road_labels = np.where(np.logical_or(np.logical_or(labels == 4, labels == 7), labels == 10), 255, 0)  # 4,7,10是道路

    image = np.repeat(np.zeros(resolution)[:, :, None], 3, axis=-1)  # 创建一个3通道的语义分割空白图
    image[index[:, 0], index[:, 1]] = np.repeat(road_labels, 3, -1)

    # Define the structure size for the dilation
    dilation_size = 3
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def pointcloud_to_image(ply_name):
    start = time.time()
    # 地理信息和分辨率
    resolution = 2560
    # 区分数据集Sensat中两地区名称
    if 'cambridge' not in ply_name.lower():  # birmingham
        resolution = 1536

    data = read_ply(ply_name)
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    # print(data.dtype.names)
    labels = np.vstack((data['label']))

    # if "test" in ply_name:
    #     labels = np.where(labels == 1, 7, 0)  # test数据中，1代表道路
    # 训练网络中若为道路，就同样标注为1
    labels = np.where(labels == 1, 7, 0)  

    points_data = np.hstack((xyz, labels))
    whole_points_data = points_data
    # whole_points_data = np.vstack((whole_points_data, points_data))
    # print("current whole input points:")
    # print(whole_points_data.shape)

    base_name = os.path.splitext(os.path.basename(ply_name))[0]
    # 构建输出图片文件的完整路径
    image_binary = points_to_image(whole_points_data, [resolution, resolution])
    print("Time Passed of pointcloud_to_image: " + str((time.time() - start)/60) + " min")
    return image_binary

