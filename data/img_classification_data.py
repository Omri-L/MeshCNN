import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
import torchvision
import torchvision.transforms as transforms

class MeshCifar10(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
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
        self.size = len(self.dataset.data)
        # self.get_mean_std()  # todo?
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = 6
        self.mesh_file = r'E:\Omri\FinalProject\QuadMesh\MeshCNN\datasets\cifar-10-batches-py\test.obj'  # TODO change

    def __getitem__(self, index):
        data = self.dataset.data[index]
        label = self.dataset.targets[index]
        mesh = Mesh(file=self.mesh_file, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder, img_data=data)
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

