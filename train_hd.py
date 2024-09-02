import importlib
import argparse
from omegaconf import OmegaConf
import os.path as osp
from datasets.inference_dataset import *
from datasets import *
import torch 

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', help='the path to the setup config file', default='cfg/train_sk.yaml')
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
cluster_cfg = OmegaConf.load(cfg.cluster_cfg)
model_cfg = OmegaConf.load(cfg.model_cfg)
cfg = OmegaConf.merge(cfg,cluster_cfg,model_cfg)

if __name__ == "__main__":
    #Get info relative to the set
    if cfg.source == "semantickitti":
        source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
        train_set = SemanticKITTI(source_data_cfg,'train')
    elif cfg.source == "nuscenes":
        source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes.yaml"))
        train_set = nuScenes(source_data_cfg,'train')
    else:
        raise  NameError('source dataset not supported')

    if cfg.target == "semantickitti":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
        train_set_2 = SemanticKITTI(target_data_cfg,'train')
        target_set = SemanticKITTI(target_data_cfg,'valid')
    elif cfg.target == "nuscenes":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes.yaml"))
        train_set_2 = nuScenes(target_data_cfg,'train')
        target_set = nuScenes(target_data_cfg,'valid')
    elif cfg.target == "semanticposs":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semanticposs.yaml"))
        train_set_2 = SemanticPOSS(target_data_cfg,'train')
        target_set = SemanticPOSS(target_data_cfg,'valid')
    elif cfg.target == "semantickitti-nuscenes":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti-nuscenes.yaml"))
        train_set_2 = SemanticKITTI_Nuscenes(target_data_cfg,'train')
        target_set = SemanticKITTI_Nuscenes(target_data_cfg,'valid')
    elif "pandaset" in cfg.target:
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,cfg.target+".yaml"))
        train_set_2 = Pandaset(target_data_cfg,'train')
        target_set = Pandaset(target_data_cfg,'valid')
    
    else:
        raise  NameError('target dataset not supported')

    #Get info relative to the model
    if cfg.architecture.model == "KPCONV":
        module = importlib.import_module('models.kpconv.kpconv')
        model_information = getattr(module, cfg.architecture.type)()
        model_information.num_classes = train_set.get_n_label()
        model_information.ignore_label = -1
        model_information.in_features_dim = model_cfg.architecture.n_features
        model_information.train_hd = cfg.train_hd
        from models.kpconv_model import SemanticSegmentationModel
        module = importlib.import_module('models.kpconv.architecture')
        model_type = getattr(module, cfg.architecture.type)
        model = SemanticSegmentationModel(model_information,cfg,model_type)
    elif cfg.architecture.model == "SPVCNN":
        module = importlib.import_module('models.spvcnn.spvcnn')
        model_information = getattr(module, cfg.architecture.type)
        model_information.num_classes = train_set.get_n_label()
        model_information.ignore_label = -1
        model_information.in_features_dim = model_cfg.architecture.n_features
        from models.spvcnn_model import SemanticSegmentationSPVCNNModel
        model = SemanticSegmentationSPVCNNModel(model_information,cfg)
    else:
        raise  NameError('model not supported')
        
    # Get HD info
    if cfg.train_hd or cfg.test_hd:
        hd_cfg = OmegaConf.load(cfg.hd_param)
        cfg = OmegaConf.merge(cfg,hd_cfg) 
        from models.HD import OnlineHD
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        model_hd = OnlineHD(hd_cfg.n_features, hd_cfg.n_dimensions, hd_cfg.n_classes, epochs = hd_cfg.epochs, device=device)
    
    # Define the path for the "HD" folder
    hd_folder = os.path.join(cfg.save_pred_path, f'HD_{hd_cfg.hd_block_stop}')

    # Check if the "HD" folder exists
    if not os.path.exists(hd_folder):
        os.makedirs(hd_folder)
        print(f"Folder 'HD' created at {hd_folder}")
    else:
        print(f"Folder 'HD' already exists at {hd_folder}")

    if cfg.train_hd:
        
        output_dataset = InferenceDataset(cfg,train_set,train_set_2, model, model_information, model_hd, target_set)
        
        output_dataset.compute_hd_dataset(hd_folder)
    
    if cfg.test_hd:

        # Define file names for saving the tensors
        weights_path = os.path.join(hd_folder, 'weights_9.pt')
        encoding_path = os.path.join(hd_folder, 'encoding_9.pt')
        
        model_hd.model.weight = torch.load(weights_path)
        model_hd.encoder.weight = torch.load(encoding_path)
        
        output_dataset_2 = InferenceDataset(cfg, train_set, target_set, model, model_information, model_hd)
        
        output_dataset_2.compute_dataset()
        ius, miu = output_dataset_2.compute_results('Pred.h5') # The results are already there?
        print(ius)
        print(miu)