import numpy as np 
import torch 
import os, time
import os.path as osp 
from utils.slam import *
from tqdm import tqdm
import h5py

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
        accumulated_confidence = accumulated_confidence.reshape((accumulated_confidence.shape[0]))
        accumulated_pointcloud = fts[:,:-1]
        accumulated_pointcloud[:,4] = lbls.reshape(-1)
        accumulated_confidence = fts[:,-1].reshape(-1,1)
        return accumulated_pointcloud.astype(np.float64), accumulated_confidence.astype(np.float64)
from cpp_wrappers.cpp_preprocess.propagation import compute_labels, cluster
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix, roc_auc_score

def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def confmat_computations_parallel(frame_full, frame_list, label_list=np.arange(-1,19),n_process=16,source_mapping=None,target_mapping=None):
    # n_process was originally 20
    global compute_conf_mat
    def compute_conf_mat(frame):
        with h5py.File(frame_full, 'r') as hdf_file:
            results = hdf_file[frame][:]
            #print(np.bincount(results[:,-1].astype(np.int32)+1))
            #print(np.bincount(results[:,0].astype(np.int32)+1))
            #print(target_mapping)
            #print(source_mapping)
            gt = results[:,-1].astype(np.int32)+1 # target_mapping[
            
            pred = results[:,0].astype(np.int32)+1 # prev source_mapping target_mapping[
            c_mat = confusion_matrix(gt,pred,labels=label_list)
            return c_mat
    
    print("The length of frame list is: ", len(frame_list)) # Check inference length
    with Pool(n_process) as p:
        all_cmat = np.array(p.map(compute_conf_mat, frame_list))
    cmat = np.sum(all_cmat,axis=0)
    return cmat

def infer_concat_kp(model,clusters,device,config_model,batch_size,config_data, hd_model=None, labels=None): # This is the model that runs the inference
    smax = torch.nn.Softmax(dim=1)
    cluster_outputs_proba = []
    for k in range(len(clusters)//batch_size+(len(clusters)%batch_size!=0)):
        l=0
        sub_clusters = clusters[k*batch_size:min((k+1)*batch_size,len(clusters))]
        r_clouds, r_inds_list = model.prepare_data(sub_clusters,False,True)
        if 'cuda' in device.type:
            r_clouds.to(device)
        outputs = smax(model.model(r_clouds, config_model, config_data, hd_model, labels)) # This are the individual predictions
        pred = outputs.detach().cpu().numpy()
        del outputs
        preds = []
        lengths = r_clouds.lengths[0].cpu().numpy()
        for i in range(len(sub_clusters)):
            L = lengths[i]
            local_cloud = pred[l:l+L]
            preds.append(local_cloud[r_inds_list[i]])
            l += L
        cluster_outputs_proba += preds
        del r_clouds
        torch.cuda.empty_cache()
    print("cluster_proba", cluster_outputs_proba[0].shape)
    return cluster_outputs_proba

def train_hd_concat_kp(model,clusters,device,config_model,batch_size,config_data, hd_model=None, labels=None): # This is the model that runs the inference
    label_whole = []
    for k in range(len(clusters)//batch_size+(len(clusters)%batch_size!=0)):
        l=0
        sub_clusters = clusters[k*batch_size:min((k+1)*batch_size,len(clusters))]
        r_clouds, r_inds_list = model.prepare_data(sub_clusters,False,True)
        if 'cuda' in device.type:
            r_clouds.to(device)
        model.model.train_hd(r_clouds, config_model, config_data, hd_model, label_whole) # This are the individual predictions
        lab = r_clouds.labels.cpu().numpy()
        labels_here = []
        lengths = r_clouds.lengths[0].cpu().numpy()
        print("Len_sub:", len(sub_clusters))
        for i in range(len(sub_clusters)):
            L = lengths[i]
            local_cloud = lab[l:l+L]
            labels_here.append(local_cloud[r_inds_list[i]])
            l += L
        label_whole += labels_here
        del r_clouds
        torch.cuda.empty_cache()
    print("Length", len(label_whole))
    if label_whole != labels.shape:
        print("False")
    
    # Check if all elements are equal
    print(len(label_whole))
    print(labels.shape)
    print(torch.equal(label_whole[0], labels))
    z = input("Enter")
    return

def infer_concat_spv(model,clusters,device,config_model,batch_size):
    smax = torch.nn.Softmax(dim=1)
    cluster_outputs_proba = []
    for k in range(len(clusters)//batch_size+(len(clusters)%batch_size!=0)):
        l=0
        sub_clusters = clusters[k*batch_size:min((k+1)*batch_size,len(clusters))]
        r_clouds, inputs_invs = model.prepare_data(sub_clusters,False)
        if 'cuda' in device.type:
            r_clouds.to(device)
        outputs = smax(model.model(r_clouds, config_model))
        _outputs = []
        for idx in range(inputs_invs.C[:, -1].max() + 1):
            cur_scene_pts = (r_clouds["lidar"].C[:, -1] == idx).cpu().numpy()
            cur_inv = inputs_invs.F[inputs_invs.C[:, -1] == idx].cpu().numpy()
            outputs_mapped = outputs[cur_scene_pts][cur_inv]
            _outputs.append(outputs_mapped)
        cluster_outputs_proba += [o.detach().cpu().numpy()for o in _outputs]
        del r_clouds
        del outputs
        del _outputs
        torch.cuda.empty_cache()
    return cluster_outputs_proba 

class InferenceDataset:

    def __init__(self, config, ref_dataset, trg_datast, model, config_model, hd_model=None, val_set=None):
        self.config = config 
        self.ref_dataset = ref_dataset 
        self.trg_datast = trg_datast 
        self.save = osp.join(self.config.save_pred_path,self.trg_datast.config.data.name) # self.config.logger.model_name
        self.n_label = self.trg_datast.get_n_label()
        self.model = model 
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            checkpoint = torch.load(os.path.join(self.config.logger.save_path,self.config.logger.model_name))
        else:
            self.device = torch.device("cpu") 
            checkpoint = torch.load(os.path.join(self.config.logger.save_path,self.config.logger.model_name), map_location=self.device)
        #checkpoint = torch.load(os.path.join(self.config.logger.save_path,"best_mIoU_"+self.config.logger.model_name), map_location=self.device)
        self.model.model.to(self.device)
        self.model.model.load_state_dict(checkpoint)
        self.model.model.eval()
        self.cfg_model = config_model
        if self.config.architecture.model == "KPCONV":
            if config.train_hd:
                self.infer_concat_train = train_hd_concat_kp
            self.infer_concat = infer_concat_kp
        elif self.config.architecture.model == "KPCONV":
            self.infer_concat = infer_concat_spv
        self.hd_model = hd_model
        if val_set:
            self.val_set = val_set

    def compute_results(self, name, file_list=None, seq=None):
        if self.config.target == "semantickitti":
            from mapping.sk import map
        elif self.config.target == "nuscenes":
            from mapping.ns import map

        if "pandaset" in self.config.target:
            mapping = map["pandaset"]
        else:
            mapping = map[self.config.target]
        labels = mapping["labels_name"]
        n_labels = len(labels)
        source_mapping = np.zeros(len(list(mapping['source_to_common'].keys()))+1) -1
        target_mapping = np.zeros(len(list(mapping['target_to_common'].keys()))+1) -1
        for key in mapping['source_to_common']:
            source_mapping[key+1] = mapping['source_to_common'][key]
        for key in mapping['target_to_common']:
            target_mapping[key+1] = mapping['target_to_common'][key]
        conf_mat = np.zeros((n_labels,n_labels))
        if file_list:
            seq = self.val_set.sequence[seq]
            #seq_path = osp.join(self.save, seq)
            #print(self.trg_datast.get_size_seq(int(0)))
            conf_mat += confmat_computations_parallel(os.path.join(self.save, f'HD_{self.config.hd_block_stop}', seq, name), file_list, np.arange(0,n_labels),20,source_mapping=source_mapping,target_mapping=target_mapping)
        else:
            for i in range(len(self.trg_datast.sequence)):
                seq = self.trg_datast.sequence[i]
                #seq_path = osp.join(self.save, seq)
                #print(self.trg_datast.get_size_seq(int(0)))
                file_list = [f'{f}' for f in range(self.trg_datast.get_size_seq(0))]
                conf_mat += confmat_computations_parallel(os.path.join(self.save, f'HD_{self.config.hd_block_stop}', seq, name), file_list, np.arange(0,n_labels),20,source_mapping=source_mapping,target_mapping=target_mapping)
        ius = per_class_iu(conf_mat)
        miu = np.nanmean(ius)
        return ius, miu

    def compute_dataset(self):
        print("Sequence: ", self.trg_datast.sequence)
        for i in tqdm(range(len(self.trg_datast.sequence)),desc="Processing dataset "+str(self.config.target)):
            self.compute_sequence(i)
            
    def compute_hd_dataset(self, hd_folder):
        print("Current Sequence: ", self.trg_datast.sequence)
        print(range(len(self.trg_datast.sequence)))
        for i in tqdm(range(len(self.trg_datast.sequence)),desc="Processing dataset "+str(self.config.target)):
            self.train_hd_sequence(i)
            # Save the tensors\
            # Define file names for saving the tensors
            weights_path = os.path.join(hd_folder, f'weights_{i}.pt')
            encoding_path = os.path.join(hd_folder, f'encoding_{i}.pt')
            torch.save(self.hd_model.model.weight, weights_path)
            torch.save(self.hd_model.encoder.weight, encoding_path)

            file_list = self.compute_small_sequence(0,i)
            ius, miu = self.compute_results(f'Pred_sm_{i}.h5', file_list, 0) # The results are already there?
            print(ius)
            print(miu)

    def compute_small_sequence(self,seq_number,training_seq):
        os.makedirs(osp.join(self.save,f'HD_{self.config.hd_block_stop}', self.val_set.sequence[seq_number]),exist_ok=True)

        #get slam poses
        rot, trans = self.val_set.get_poses_seq(seq_number)

        #get sequence information
        len_seq = 10
        seq = self.val_set.sequence[seq_number]
        
        start = [i for i in range(self.config.subsample)]
        file_list = []

        with h5py.File(osp.join(self.save, f'HD_{self.config.hd_block_stop}', seq, f'Pred_sm_{training_seq}.h5'), 'w') as hdf_file:
            for st in start:
                for frame in tqdm(range(st,len_seq,len(start)),leave=False,desc="Sequence: " + str(self.trg_datast.sequence[seq_number]) + ", subsample number " +str(st+1)+"/"+str(len(start))): # len_seq   
                #for frame in range(st,len_seq,len(start)):    
                                   
                    pointcloud, label = self.trg_datast.loader(seq,frame)

                    local_rot, local_trans = rot[frame], trans[frame]

                    #add channel for semantic and for timestamp
                    pointcloud = np.hstack((pointcloud[:,:4],np.zeros(len(pointcloud)).reshape(-1,1)-1,np.zeros(len(pointcloud)).reshape(-1,1)+frame)) # np.zeros(len(pointcloud)).reshape(-1,1)-1
                    pointcloud = apply_transformation(pointcloud, (local_rot, local_trans))

                    #clusters = cluster(pointcloud, label, len(pointcloud), self.config.cluster.voxel_size, self.config.cluster.n_centroids, 'Kmeans')
                    #clusters = list(filter(lambda e: len(e)>1,clusters))
                    #clusters = [np.array(c) for c in clusters]

                    # Predictions are made here :0
                    total_pred = self.infer_concat(self.model,[pointcloud],self.device,self.cfg_model, 8, self.config, self.hd_model, label)
                    #predicted_cloudwise = np.zeros((len(pointcloud),self.n_label+1))
                    #for i in range(len(clusters)):
                    #    predicted_cloudwise[clusters[i],1:self.n_label+1] = np.maximum(total_pred[i],predicted_cloudwise[clusters[i],1:self.n_label+1])

                    pred = np.argmax(total_pred[0][:,:self.n_label+1], axis=1) -1

                    to_save = np.zeros((len(pointcloud),2),dtype=np.int32)
                    to_save[:,0] = pred.astype(np.int32)
                    to_save[:,-1] = label.astype(np.int32)
                    hdf_file.create_dataset(f'{str(frame)}', data=to_save)
                    file_list.append(f'{str(frame)}')
                
                    #except OSError as e: # Values are not found
                    #    continue
        
        return file_list

    def compute_sequence(self,seq_number):
        if osp.exists(osp.join(self.save,f'HD_{self.config.hd_block_stop}',self.trg_datast.sequence[seq_number], 'Pred.h5')): # This was commented to reduce the amount of times that the data is calculated
            print("Skip")
            print(osp.join(self.save,self.trg_datast.sequence[seq_number]))
            return True 
        os.makedirs(osp.join(self.save,f'HD_{self.config.hd_block_stop}',self.trg_datast.sequence[seq_number]),exist_ok=True)

        #init accumulated arrays
        accumulated_pointcloud = np.empty((0,6))
        accumulated_confidence = np.empty(0, dtype=np.float)

        #get slam poses
        rot, trans = self.trg_datast.get_poses_seq(seq_number)

        #get sequence information
        len_seq = self.trg_datast.get_size_seq(seq_number)
        seq = self.trg_datast.sequence[seq_number]
        
        #accumulate
        lastIndex = 1
        local_limit = self.config.sequence.limit_GT_time
        start = [i for i in range(self.config.subsample)]
        
        #len_seq = 1000
        #st_real = False # Cariable that makes the the whole loop run at least once

        print(osp.join(self.save, f'HD_{self.config.hd_block_stop}', seq, 'Pred.h5'))
        with h5py.File(osp.join(self.save, f'HD_{self.config.hd_block_stop}', seq, 'Pred.h5'), 'w') as hdf_file:
            for st in start:
                for frame in tqdm(range(st,len_seq,len(start)),leave=False,desc="Sequence: " + str(self.trg_datast.sequence[seq_number]) + ", subsample number " +str(st+1)+"/"+str(len(start))): # len_seq   
                #for frame in range(st,len_seq,len(start)):                   
                    pointcloud, label = self.trg_datast.loader(seq,frame)

                    #if st_real: # frame>st
                        #raise Exception("Just one for now") # This needs to be removed
                        #Check if the sensor moved more than min_dist_mvt
                    #    if np.linalg.norm(local_trans - trans[frame-lastIndex]) < self.config.sequence.min_dist_mvt:
                    #        accumulated_pointcloud = accumulated_pointcloud[:-len(pointcloud)]
                    #        accumulated_confidence = accumulated_confidence[:-len(pointcloud)]
                    #        local_limit += 1
                    #        lastIndex += 1
                    #    else:
                    #        lastIndex =1

                        #voxelize the past sequence and remove old points
                    #    if len(accumulated_pointcloud) > 0:
                    #        accumulated_pointcloud, accumulated_confidence = grid_subsample(accumulated_pointcloud, accumulated_confidence, self.config.sequence.subsample)
                    #        accumulated_confidence = accumulated_confidence[accumulated_pointcloud[:,-1] > frame - local_limit]
                    #        accumulated_pointcloud = accumulated_pointcloud[accumulated_pointcloud[:,-1] > frame - local_limit]

                    # norm_curr = np.linalg.norm(pointcloud[:,:3],axis=1)
                    # pointcloud = pointcloud[norm_curr>0]
                    # label = label[norm_curr>0]
                    # norm_curr = np.linalg.norm(pointcloud[:,:3],axis=1)
                    # pointcloud = pointcloud[norm_curr<75]
                    # label = label[norm_curr<75]

                    local_rot, local_trans = rot[frame], trans[frame]

                    #add channel for semantic and for timestamp
                    pointcloud = np.hstack((pointcloud[:,:4],np.zeros(len(pointcloud)).reshape(-1,1)-1,np.zeros(len(pointcloud)).reshape(-1,1)+frame)) # np.zeros(len(pointcloud)).reshape(-1,1)-1
                    pointcloud = apply_transformation(pointcloud, (local_rot, local_trans))
                    print("pointcloud: ", pointcloud.shape)

                    #remove accumulated points too far from the center
                    #if len(accumulated_pointcloud)>0:
                    #    center_current = np.mean(pointcloud[:,:2],axis=0)
                    #    norm_acc = np.linalg.norm(accumulated_pointcloud[:,:2]-center_current,axis=1)
                    #    accumulated_pointcloud = accumulated_pointcloud[norm_acc<self.config.sequence.limit_GT]
                    #    accumulated_confidence = accumulated_confidence[norm_acc<self.config.sequence.limit_GT]

                    #accumulated_pointcloud = np.vstack((accumulated_pointcloud,pointcloud))
                    #accumulated_confidence = accumulated_confidence.reshape((accumulated_confidence.shape[0]))
                    #accumulated_confidence = np.concatenate((accumulated_confidence,np.zeros(len(pointcloud))))

                    #acc_label = np.copy(accumulated_pointcloud[:,4].astype(np.int32))
                    #acc_label, new_conf = compute_labels(accumulated_pointcloud, acc_label, accumulated_confidence, len(pointcloud), self.config.sequence.voxel_size, self.n_label, self.config.source, self.config.sequence.dist_prop)

                    #dynamic_indices = np.where(self.ref_dataset.get_dynamic(acc_label))[0]
                    #dynamic_current = dynamic_indices[dynamic_indices > (len(acc_label) - len(label))]
                    #acc_label[dynamic_current] = -1
                    #new_conf[dynamic_current] = 0

                    #clusters = cluster(accumulated_pointcloud, acc_label, len(pointcloud), self.config.cluster.voxel_size, self.config.cluster.n_centroids, 'Kmeans')
                    #clusters = cluster(pointcloud, label, len(pointcloud), self.config.cluster.voxel_size, self.config.cluster.n_centroids, 'Kmeans')
                    #clusters = list(filter(lambda e: len(e)>1,clusters))
                    #clusters = [np.array(c) for c in clusters]

                    # Predictions are made here :0
                    #total_pred = self.infer_concat(self.model,[accumulated_pointcloud[c] for c in clusters],self.device,self.cfg_model, 8, self.config, self.hd_model, label) # Change 1 to change the batch size
                    total_pred = self.infer_concat(self.model,[pointcloud],self.device,self.cfg_model, 8, self.config, self.hd_model, label)
                    #predicted_cloudwise = np.zeros((len(accumulated_pointcloud),self.n_label+1))
                    #predicted_cloudwise = np.zeros((len(pointcloud),self.n_label+1))
                    #for i in range(len(clusters)):
                    #    predicted_cloudwise[clusters[i],1:self.n_label+1] = np.maximum(total_pred[i],predicted_cloudwise[clusters[i],1:self.n_label+1])

                    pred = np.argmax(total_pred[0][:,:self.n_label+1], axis=1)-1
                    #score = np.max(predicted_cloudwise[:,1:self.n_label+1],axis=1)
                    #comp_score = np.copy(score)
                    #comp_score[comp_score>self.config.cluster.override] = 1 

                    #print("Accumulated PointCloud:", accumulated_pointcloud.shape)
                    #print("Pointcloud: ", pointcloud.shape) # Actually incoming
                    #print("Pred:", pred.shape)

                    #accumulated_pointcloud[-len(pointcloud):,4] = np.where(new_conf[-len(pointcloud):]>comp_score[-len(pointcloud):],acc_label[-len(pointcloud):],pred[-len(pointcloud):])
                    #accumulated_confidence[-len(pointcloud):] = np.where(new_conf[-len(pointcloud):]>comp_score[-len(pointcloud):],new_conf[-len(pointcloud):],score[-len(pointcloud):])

                    to_save = np.zeros((len(pointcloud),2),dtype=np.int32)
                    #to_save[:,0] = accumulated_pointcloud[-len(pointcloud):,4].astype(np.int32)
                    to_save[:,0] = pred.astype(np.int32)
                    to_save[:,-1] = label.astype(np.int32)
                    #np.save(osp.join(self.save, f'HD_{self.config.hd_block_stop}', seq, str(frame)+'.npy'), to_save)
                    hdf_file.create_dataset(f'{str(frame)}', data=to_save)

                    #st_real = True
                    
    def train_hd_sequence(self,seq_number):

        #init accumulated arrays
        accumulated_pointcloud = np.empty((0,6))
        accumulated_confidence = np.empty(0, dtype=np.float)

        #get slam poses
        rot, trans = self.trg_datast.get_poses_seq(seq_number)

        #get sequence information
        len_seq = self.trg_datast.get_size_seq(seq_number)
        seq = self.trg_datast.sequence[seq_number]
        
        #accumulate
        lastIndex = 1
        local_limit = self.config.sequence.limit_GT_time
        start = [i for i in range(self.config.subsample)]
        #len_seq = 1000
        st_real = False
        reset = 0
        for st in start:
            for frame in tqdm(range(st,len_seq,len(start)),leave=False,desc="Sequence: " + str(self.trg_datast.sequence[seq_number]) + ", subsample number " +str(st+1)+"/"+str(len(start))): # len_seq                
                try:                    
                    pointcloud, label = self.trg_datast.loader(seq,frame)
                    
                    if st_real: # 
                        #raise Exception("Just one for now") # This needs to be removed
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

                    local_rot, local_trans = rot[frame], trans[frame]

                    #add channel for semantic and for timestamp
                    pointcloud = np.hstack((pointcloud[:,:4],label.reshape(-1,1),np.zeros(len(pointcloud)).reshape(-1,1)+frame)) # np.zeros(len(pointcloud)).reshape(-1,1)-1
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
                    acc_label, new_conf = compute_labels(accumulated_pointcloud, acc_label, accumulated_confidence, len(pointcloud), self.config.sequence.voxel_size, self.n_label, self.config.source, self.config.sequence.dist_prop)


                    dynamic_indices = np.where(self.ref_dataset.get_dynamic(acc_label))[0]
                    dynamic_current = dynamic_indices[dynamic_indices > (len(acc_label) - len(label))]
                    acc_label[dynamic_current] = -1
                    new_conf[dynamic_current] = 0

                    clusters = cluster(accumulated_pointcloud, acc_label, len(pointcloud), self.config.cluster.voxel_size, self.config.cluster.n_centroids, 'Kmeans')
                    clusters = list(filter(lambda e: len(e)>1,clusters))
                    clusters = [np.array(c) for c in clusters]

                    # Predictions are made here :0

                    self.infer_concat_train(self.model,[accumulated_pointcloud[c] for c in clusters],self.device,self.cfg_model, 8 , self.config, self.hd_model, label) # Change 1 to change the batch size

                    st_real = True

                    if reset % 10 == 9:
                        accumulated_pointcloud = np.empty((0,6))
                        accumulated_confidence = np.empty(0, dtype=np.float)
                        lastIndex = 1
                        local_limit = self.config.sequence.limit_GT_time
                        print("Reseted :0")
                        #x = input()
                    
                    reset += 1

                except OSError as e: # Values are not found
                    
                    continue             

