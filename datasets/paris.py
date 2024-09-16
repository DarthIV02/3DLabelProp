from datasets.base_dataset import PointCloudDataset
import os.path as osp 
import os
import numpy as np 
from utils.slam import *
from auxiliary.ply_utils import *

class Paris(PointCloudDataset):

    def __init__(self, config, split, dynamic=[0]):
        self.split = split
        self.config = config
        learning_map = config.learning_map
        label_map_names = config.labels
        self.map = np.zeros(max(learning_map.keys())+1)
        self.label_names = {}
        for key in learning_map:
            self.map[key] = learning_map[key]
            if learning_map[key] not in self.label_names and learning_map[key] !=0:
                self.label_names[learning_map[key]] = label_map_names[key]
        self.label_names = list(self.label_names.values())
        #self.sequence = []
        self.sequence = ["0"]
        self.path = config.data.path
        self.traj_folder = config.data.traj_folder
        self.dynamic = np.array(dynamic)
        self.color_map = config.color_map

    def loader(self, seq, frame):
        seq_path = osp.join(self.path,str(seq))
        pointcloud_label = osp.join(seq_path+'.ply')
        labels_read = read_ply(pointcloud_label)['class']
        sem_label = (labels_read & 0xFFFF)
        pointcloud = read_ply(pointcloud_label)[['x', 'y', 'z']]
        sem_label = sem_label.astype(np.int32)
        return pointcloud, self.map_to_eval(sem_label)

    def map_to_eval(self, og_labels):
        #labels between -1 and n_label-1
        return self.map[og_labels].astype(np.int32) - 1

    def get_n_label(self):
        return int(np.max(self.map))

    def get_dynamic(self, labels):
        belonging = np.zeros_like(labels)
        for id in self.dynamic:
            belonging = np.logical_or(belonging,labels==id)
        return belonging

    def get_static(self, labels):
        return np.logical_and(np.logical_not(self.get_dynamic(labels)),labels>-1)

    def get_sequence(self,seq_number):
        return list(os.listdir(osp.join(self.path,self.sequence[seq_number]))).sort()

    def get_size_seq(self, seq_number):
        last = os.listdir(osp.join(self.path,self.sequence[seq_number]))
        last.sort()
        file = last[-1]
        print("Last: ", file)
        file = file[:-4]
        print("Last: ", file)
        return int(file)

    def get_poses_seq(self, seq_number):
        return read_transfo(osp.join(osp.join(self.path,self.traj_folder),self.sequence[seq_number]+'_traj.txt'),False)