import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random


class MeshCifar10(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.mesh_file = r'E:\Omri\FinalProject\QuadMesh\MeshCNN\datasets\cifar-10-batches-py\test.obj'  # TODO change
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot

        self.train = False
        if self.opt.phase == 'train':
            self.train = True

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train,
                                                download=True,
                                                transform=transform)

        self.dir = os.path.join(opt.dataroot)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        # self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes)

        self.dataset_reduction(1)

        self.size = len(self.dataset.data)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def dataset_reduction(self, factor):
        if factor == 1:
            return

        full_dataset_size = len(self.dataset.data)
        desired_dataset_size = int(np.ceil(factor * full_dataset_size))

        all_indices = []

        np.random.seed(0)
        random.seed(0)

        for c in self.dataset.class_to_idx:
            c_ind = self.dataset.class_to_idx[c]
            indices = np.where(np.array(self.dataset.targets) == c_ind)[0]
            indices = (np.random.choice(indices,
                             int(np.ceil(desired_dataset_size / len(self.dataset.classes))),
                             replace=False)).tolist()
            all_indices = all_indices + indices

        random.shuffle(all_indices)
        self.dataset.targets = [self.dataset.targets[i] for i in all_indices]
        self.dataset.data = [self.dataset.data[i] for i in all_indices]

        np.random.seed()
        random.seed()
        return

    def __getitem__(self, index):
        data = self.dataset.data[index]
        label = self.dataset.targets[index]
        mesh = Mesh(file=self.mesh_file, opt=self.opt, hold_history=False,
                    export_folder=self.opt.export_folder, img_data=data,
                    img_ind=index)
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

