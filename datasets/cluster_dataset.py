import numpy as np 
import torch 
import os 
import os.path as osp 
from utils.slam import *
try:
    from torchsparse.utils.quantize import sparse_quantize
    def grid_subsample(accumulated_pointcloud,accumulated_confidence,vox_size):
        _, indices = sparse_quantize(accumulated_pointcloud[:,:3], vox_size,return_index=True)
        accumulated_pointcloud = accumulated_pointcloud[indices]
        accumulated_confidence = accumulated_confidence[indices]
        return accumulated_pointcloud,accumulated_confidence
except:
    import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
    def grid_subsample(accumulated_pointcloud, accumulated_confidence, vox_size):
        accumulated_confidence = accumulated_confidence.reshape((accumulated_confidence.shape[0], 1))
        _, fts, lbls = cpp_subsampling.subsample(accumulated_pointcloud[:,:3].astype(np.float32),
                                                features=np.hstack((accumulated_pointcloud,accumulated_confidence)).astype(np.float32),
                                                classes = accumulated_pointcloud[:,4].astype(np.int32).reshape(-1,1),
                                                sampleDl=vox_size,
                                                verbose=False)
        accumulated_pointcloud = fts[:,:-1]
        accumulated_pointcloud[:,4] = lbls.reshape(-1)
        accumulated_confidence = fts[:,-1].reshape(-1,1)
        return accumulated_pointcloud.astype(np.float64), accumulated_confidence.astype(np.float64)
from cpp_wrappers.cpp_preprocess.propagation import compute_labels, cluster
import random
from tqdm import tqdm

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class BalancedSampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source,n_label,shuffle=50000,batch_size=1):
        self.data_source = data_source #list of lists
        self.n_label = n_label
        self.reshuffle = shuffle
        self.batch_size=batch_size
        try:
            self.data_source = [ds.tolist() for ds in data_source]
        except:
            self.data_source = data_source

    def __iter__(self):
        lbl_curr = 0
        counter = [0 for _ in range(self.n_label)]
        c = 0
        self.data_source = [random.sample(ds,len(ds)) for ds in self.data_source]
        batch = []
        for _ in range(sum([len(ds) for ds in self.data_source])):
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            batch.append(int(self.data_source[lbl_curr][counter[lbl_curr]]))
            counter[lbl_curr] = (counter[lbl_curr] + 1) % len(self.data_source[lbl_curr])
            if counter[lbl_curr] == 0:
                self.data_source[lbl_curr] = random.sample(self.data_source[lbl_curr],len(self.data_source[lbl_curr]))
            lbl_curr = (lbl_curr + 1) % self.n_label
            c += 1
            if c == self.reshuffle:
                self.data_source = [random.sample(ds,len(ds)) for ds in self.data_source]
                counter = [0 for _ in range(self.n_label)]
                c = 0

    def __len__(self):
        raise NotImplementedError

class ClusterDataset(torch.utils.data.Dataset):

    def __init__(self, config, dataset):
        self.config = config 
        self.dataset = dataset 
        self.cluster_path = osp.join(self.config.cluster.path,self.config.source,self.config.cluster.name)
        self.n_label = self.dataset.get_n_label()
        self.generate_dataset()
        print("Cluster dataset "+self.dataset.split+" created.")
        self.get_weight()
        print("Data information for "+self.dataset.split+" created. Loader ready.")

    def __len__(self):
        return len(self.datalist) 

    def get_class_names(self):
        return self.dataset.label_names

    def __getitem__(self, index):
        index = int(self.datalist[index,-1])
        seq = index//(self.n_clust_max*self.size_seq_max)
        scan = (index-seq*self.n_clust_max*self.size_seq_max)//self.n_clust_max
        cluster_number = index%self.n_clust_max
        return self.get_cluster(seq,scan,cluster_number)

    def get_cluster(self,seq_number,frame_number,cluster_number):
        try:
            return np.fromfile(osp.join(self.cluster_path,self.dataset.sequence[seq_number],str(frame_number)+'_'+str(cluster_number)+'.bin'),dtype=np.float32).reshape(-1,6)
        except:
            #return None
            raise NameError("Cluster : " + str(self.dataset.sequence[seq_number]) + "_" + str(frame_number)+ "_"  + str(cluster_number) + " not found")

    def get_dataset(self):
        self.size_seq_max = max([self.dataset.get_size_seq(s) for s in range(len(self.dataset.sequence))])
        self.n_clust_max = self.config.cluster.n_centroids
        self.total = 0
        for seq in self.dataset.sequence:
            seq_path = osp.join(self.cluster_path, seq) 
            self.total += len(list(os.listdir(seq_path)))


    def get_weight(self):
        self.init_weight()
        self.class_frames = []
        self.datalist = self.datalist[np.logical_and(np.sum(self.datalist[:,:-1],axis=1)<20000,np.sum(self.datalist[:,:-1],axis=1)>100)]
        for i in range(self.n_label):
            integer_inds = np.where(self.datalist[:, i]>10)[0]
            self.class_frames.append(integer_inds.astype(np.int64))
            self.class_frames[i] = np.random.permutation(self.class_frames[i])

        class_proportions = np.sum(self.datalist[:,:-1],axis=0)
        self.w = 1/(100*class_proportions/np.sum(class_proportions))
        self.w[self.w<0.05] = 0.05
        self.w[self.w>50] = 50


    def init_weight(self):
        seq_stat_file = osp.join(self.cluster_path, self.dataset.split,'weight_stats_cluster.npy')
        if osp.exists(seq_stat_file):
            with open(seq_stat_file, 'rb') as f:
                self.datalist = np.load(f)
                self.total = len(self.datalist)
                self.n_clust_max = self.config.cluster.n_centroids
                self.size_seq_max = max([self.dataset.get_size_seq(s) for s in range(len(self.dataset.sequence))]) 
        else:
            self.get_dataset()
            self.datalist = np.zeros((self.total, self.n_label+1))
            os.makedirs(osp.join(self.cluster_path, self.dataset.split),exist_ok=True)
            i = 0
            for s in range(len(self.dataset.sequence)):
                for k in range(self.dataset.get_size_seq(s)):
                    for l in range(self.config.cluster.n_centroids):
                        clust = self.get_cluster(s,k,l)
                        if clust is None:
                            continue
                        unique, counts = np.unique(clust[clust[:,4] != -1,4], return_counts=True)
                        idx = self.n_clust_max*self.size_seq_max*s + self.n_clust_max*k + l
                        self.datalist[i,unique.astype(np.int32)] = counts
                        self.datalist[i,-1] = idx
                        i+=1
            np.save(seq_stat_file, self.datalist)

    def generate_dataset(self):
        workers = 7
        worker_id = 1
        ranges = range(len(self.dataset.sequence))
        ranges = np.array_split(ranges, workers)
        print("Starting: ", ranges[worker_id][0])
        print("Ending: ", ranges[worker_id][-1])
        x = input("Enter")
        for i in tqdm(ranges[worker_id],desc="Processing dataset "+str(self.config.source)):
            self.generate_sequence(i)
        x = input("Enter")

    def generate_sequence(self,seq_number):
        if osp.exists(osp.join(self.cluster_path,self.dataset.sequence[seq_number])):
            return True 
        os.makedirs(osp.join(self.cluster_path,self.dataset.sequence[seq_number]),exist_ok=True)

        #init accumulated arrays
        accumulated_pointcloud = np.empty((0,6)).astype(np.float64)
        accumulated_confidence = np.empty(0, dtype=np.float64)

        #get slam poses
        rot, trans = self.dataset.get_poses_seq(seq_number)

        #get sequence information
        len_seq = self.dataset.get_size_seq(seq_number)
        seq = self.dataset.sequence[seq_number]
        
        #accumulate
        lastIndex = 1
        local_limit = self.config.sequence.limit_GT_time
        start = [i for i in range(self.config.subsample)]
        for st in start:
            for frame in tqdm(range(st,len_seq,len(start)),leave=False,desc="Sequence: " + str(self.dataset.sequence[seq_number]) + ", subsample number " +str(st+1)+"/"+str(len(start))):
                if frame>st:
                    #Check if the sensor moved more than min_dist_mvt
                    if np.linalg.norm(local_trans - trans[frame-lastIndex]) < self.config.sequence.min_dist_mvt:
                        accumulated_pointcloud = accumulated_pointcloud[:-len(pointcloud)]
                        accumulated_confidence = accumulated_confidence[:-len(pointcloud)]
                        local_limit += 1
                        lastIndex += 1
                    else:
                        lastIndex =1

                    #voxelize the past sequence and remove old points
                    if len(accumulated_pointcloud) > 0:
                        accumulated_pointcloud, accumulated_confidence = grid_subsample(accumulated_pointcloud, accumulated_confidence, self.config.sequence.subsample)
                        accumulated_confidence = accumulated_confidence[accumulated_pointcloud[:,-1] > frame - local_limit]
                        accumulated_pointcloud = accumulated_pointcloud[accumulated_pointcloud[:,-1] > frame - local_limit]

                pointcloud, label = self.dataset.loader(seq,frame)

                #discard information beyond a certain distance
                if self.config.sequence.out_lim>0:
                    norm_curr = np.linalg.norm(pointcloud[:,:3],axis=1)
                    pointcloud = pointcloud[norm_curr<self.config.sequence.out_lim]
                    label = label[norm_curr<self.config.sequence.out_lim]

                local_rot, local_trans = rot[frame], trans[frame]

                #add channel for semantic and for timestamp
                pointcloud = np.hstack((pointcloud[:,:4],np.zeros(len(pointcloud)).reshape(-1,1)-1,np.zeros(len(pointcloud)).reshape(-1,1)+frame))
                pointcloud = apply_transformation(pointcloud, (local_rot, local_trans))

                #remove accumulated points too far from the center
                if len(accumulated_pointcloud)>0:
                    center_current = np.mean(pointcloud[:,:2],axis=0)
                    norm_acc = np.linalg.norm(accumulated_pointcloud[:,:2]-center_current,axis=1)
                    accumulated_pointcloud = accumulated_pointcloud[norm_acc<self.config.sequence.limit_GT]
                    accumulated_confidence = accumulated_confidence[norm_acc<self.config.sequence.limit_GT]

                accumulated_pointcloud = np.vstack((accumulated_pointcloud,pointcloud))
                accumulated_confidence = accumulated_confidence.reshape((accumulated_confidence.shape[0]))
                accumulated_confidence = np.concatenate((accumulated_confidence,np.zeros(len(pointcloud))))

                acc_label = np.copy(accumulated_pointcloud[:,4].astype(np.int32))
                acc_label, accumulated_confidence = compute_labels(accumulated_pointcloud, acc_label, accumulated_confidence, len(pointcloud), self.config.sequence.voxel_size, self.n_label, self.config.source, self.config.sequence.dist_prop)


                dynamic_indices = np.where(self.dataset.get_dynamic(acc_label))[0]
                dynamic_current = dynamic_indices[dynamic_indices > (len(acc_label) - len(label))]
                acc_label[dynamic_current] = -1

                clusters = cluster(accumulated_pointcloud, acc_label, len(pointcloud), self.config.cluster.voxel_size, self.config.cluster.n_centroids, 'Kmeans')

                clusters = list(filter(lambda e: len(e)>1,clusters))
                clusters = [np.array(c) for c in clusters]
                accumulated_pointcloud[:,4] = acc_label
                pointcloud[:,4] = accumulated_pointcloud[-len(pointcloud):,4]

                accumulated_pointcloud[-len(pointcloud):,4] = label
                accumulated_confidence[-len(pointcloud):] = 1
                #save
                for c in range(len(clusters)):
                    accumulated_pointcloud[clusters[c]].astype(np.float32).tofile(osp.join(self.cluster_path,seq,str(frame)+'_'+str(c)+'.bin'),format='float32')

