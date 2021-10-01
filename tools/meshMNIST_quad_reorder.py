import numpy as np
import os
import shutil
from tqdm import tqdm


train_dataset_path = r'E:\Omri\FinalProject\QuadMesh\meshMNIST\Quad_InstantMeshes\train_meshMNIST_100V_quad'
test_dataset_path = r'E:\Omri\FinalProject\QuadMesh\meshMNIST\Quad_InstantMeshes\t10k_meshMNIST_100V_quad'
output_path = r'E:\Omri\FinalProject\QuadMesh\meshMNIST\Quad_InstantMeshes\meshMNIST_100V_quad_new'

if not os.path.exists(output_path):
    os.mkdir(output_path)

train_labelPath = os.path.join(train_dataset_path, 'labels.txt')
train_labels = np.asarray(np.loadtxt(train_labelPath, delimiter=','),
                          dtype=np.int32)
unique_labels = set(list(train_labels[:, 1]))

test_labelPath = os.path.join(test_dataset_path, 'labels.txt')
test_labels = np.asarray(np.loadtxt(test_labelPath, delimiter=','),
                         dtype=np.int32)
unique_labels = unique_labels.union(set(test_labels[:, 1]))

for l in unique_labels:
    l_path = os.path.join(output_path, str(l))
    if not os.path.exists(l_path):
        os.mkdir(l_path)
        os.mkdir(os.path.join(l_path, 'train'))
        os.mkdir(os.path.join(l_path, 'test'))

train_objs_path = [os.path.join(dp, f) for dp, dn, filenames in os.walk(train_dataset_path)
             for f in filenames if os.path.splitext(f)[1] == '.obj']

test_objs_path = [os.path.join(dp, f) for dp, dn, filenames in os.walk(test_dataset_path)
             for f in filenames if os.path.splitext(f)[1] == '.obj']

pbar = tqdm(total=len(test_labels)+len(train_labels))
for test_label in test_labels:
    obj_num = str(test_label[0])
    obj_num = obj_num.zfill(5)
    l = str(test_label[1])
    obj_path = os.path.join(test_dataset_path, obj_num + '_instantmesh_quad.obj')
    copy_obj_path = os.path.join(output_path, str(l), 'test', obj_num + '_quad.obj')
    if not os.path.exists(copy_obj_path):
        shutil.copyfile(obj_path, copy_obj_path)
    pbar.update(1)

for train_label in train_labels:
    obj_num = str(train_label[0])
    obj_num = obj_num.zfill(6)
    l = str(train_label[1])
    obj_path = os.path.join(train_dataset_path, obj_num + '_instantmesh_quad.obj')
    copy_obj_path = os.path.join(output_path, str(l), 'train', obj_num + '_quad.obj')
    if not os.path.exists(copy_obj_path):
        shutil.copyfile(obj_path, copy_obj_path)
    pbar.update(1)

pbar.close()

