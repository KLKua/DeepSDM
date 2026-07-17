import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import rasterio
from TaxaDataset_smoothviz_prediction import TaxaDataset_smoothviz
from TaxaDataset import TaxaDataset
import pytorch_lightning as pl
import yaml
from types import SimpleNamespace
import socket
import h5py

class LitDeepSDMData(pl.LightningDataModule):
    def __init__(self, 
                 device, 
                 yaml_conf = './DeepSDM_conf.yaml', 
                 tmp_path = './tmp', 
                 ):
        
        super().__init__()
        self.device = device
        self.tmp_path = tmp_path
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)

        with open(yaml_conf, 'r') as f:
            DeepSDM_conf = yaml.load(f, Loader = yaml.FullLoader)
        DeepSDM_conf = SimpleNamespace(**DeepSDM_conf)
        
        geo_extent_file = DeepSDM_conf.geo_extent_file
        meta_json_files = DeepSDM_conf.meta_json_files # {'env_inf': './workspace/env_information.json', 'sp_inf': './workspace/species_information.json', 'k_inf': './workspace/k_information.json', 'co_vec': './workspace/cooccurrence_vector.json'}
        
        # define self.env_inf, self.sp_inf, self.co_vec
        for item_ in meta_json_files.items():
            print(f'{item_[1]}')
            with open(item_[1]) as f:
                setattr(self, item_[0], json.load(f))

        with rasterio.open(geo_extent_file) as f:
            self.geo_extent = ToTensor()(f.read(1))
            self.geo_transform = f.transform
            self.geo_crs = f.crs
        
        self.DeepSDM_conf = DeepSDM_conf
        self.training_conf = SimpleNamespace(**DeepSDM_conf.training_conf)
        self.hostname = socket.gethostname()

    def _load_env_list(self, stage_date_list, stage_env_list, stage_species_list):
        # env
        ### The order is date x env
        env_stack = {
            'date' : stage_date_list,
            'env' : stage_env_list,
        }
        env_tensor_list = []
        for date_ in stage_date_list:
            date_env_list = []
            y_m = '-'.join(date_.split('-')[:-1])
            for env_ in stage_env_list:
                with rasterio.open(os.path.join(self.env_inf['dir_base'], f"{self.env_inf['info'][env_][date_]['tif_span_avg']}")) as f:
                    img_ = ToTensor()(f.read(1)).to(self.device)
                fill = torch.full_like(img_, float(self.env_inf['info'][env_]['mean']))
                img = img_.where(self.geo_extent.to(self.device) == 1, fill)
                
                # environment factors which should be normalized
                if env_ not in self.training_conf.non_normalize_env_list:
                    img_norm = (img - self.env_inf['info'][env_]['mean']) / self.env_inf['info'][env_]['sd']
                else:
                    img_norm = img
                date_env_list.append(img_norm.to(self.device))
            env_tensor_list.append(torch.cat(date_env_list)[None, ].to(self.device))
        env_stack['tensor'] = torch.cat(env_tensor_list).to(self.device)  # env_stack['tensor'].shape = (len(stage_date_list), len(env_list), height, width)

        return env_stack
    
    def _load_label_list(self, stage_date_list, stage_env_list, stage_species_list):
        # label
        ########### WARNING, check later if the label order is aligned with the env order
        ### The order is species x date
        label_stack = {
            'species_date' : [],
            'species' : [],
            'date' : [],
        }
        label_tensor_list = []
        for species_ in stage_species_list:
            with h5py.File(os.path.join(self.sp_inf['dir_base'], f"{self.sp_inf['file_name'][species_]['h5file_name']}"), 'r') as h5f:
                for date_ in stage_date_list:
                    label_stack['species_date'].append(f'{species_}_{date_}')
                    label_stack['species'].append(f'{species_}')
                    label_stack['date'].append(f'{date_}')
                    if species_ in self.sp_inf['file_name']:
    #                     with rasterio.open(os.path.join(self.sp_inf['dir_base'], f"{self.sp_inf['file_name'][species_][date_]}")) as f:
    #                         label_tensor_list.append(ToTensor()(f.read(1)).to(self.device))
                        label_tensor_list.append(ToTensor()(h5f[self.sp_inf['file_name'][species_]['h5file_dataset_name'][date_]][:]).to(self.device))
                    else:
                        label_tensor_list.append(torch.zeros([1, 
                                                              self.DeepSDM_conf.spatial_conf_tmp['grid_size'] * self.DeepSDM_conf.spatial_conf_tmp['num_of_grid_y'], 
                                                              self.DeepSDM_conf.spatial_conf_tmp['grid_size'] * self.DeepSDM_conf.spatial_conf_tmp['num_of_grid_x']]).to(self.device))
        t = torch.cat(label_tensor_list).to(self.device)
        label_stack['tensor'] = torch.where(t < 0, 0, t).to(self.device)
        
        return label_stack

    def _load_k2_list(self, stage_date_list, stage_env_list, stage_species_list):        
        # k2 value
        k2_stack = {
            'date' : stage_date_list,
        }
        k2_tensor_list = []
        for date_ in stage_date_list:
            with rasterio.open(os.path.join(self.k_inf['dir_base'], f"{self.k_inf['file_name'][date_]}")) as f:
                k2_tensor_list.append(ToTensor()(f.read(1)).to(self.device))
        k2_stack['tensor'] = torch.cat(k2_tensor_list).to(self.device)
        
        return k2_stack

    def _load_embedding_list(self, stage_date_list, stage_env_list, stage_species_list):        
        # embeddings
        embedding = dict()
        for species_ in stage_species_list:
            embedding[species_] = self.co_vec[species_]
            
        return embedding
            
    def _load_meta_list(self, stage_date_list, stage_env_list, stage_species_list, stage):
        torch.save(self._load_env_list(stage_date_list, stage_env_list, stage_species_list), os.path.join(self.tmp_path, f'env_stack_{stage}_{self.device}_{self.hostname}.pth'))
        torch.save(self._load_embedding_list(stage_date_list, stage_env_list, stage_species_list), os.path.join(self.tmp_path, f'embedding_{stage}_{self.device}_{self.hostname}.pth'))
        torch.save(self._load_label_list(stage_date_list, stage_env_list, stage_species_list), os.path.join(self.tmp_path, f'label_stack_{stage}_{self.device}_{self.hostname}.pth'))
        torch.save(self._load_k2_list(stage_date_list, stage_env_list, stage_species_list), os.path.join(self.tmp_path, f'k2_stack_{stage}_{self.device}_{self.hostname}.pth'))


    def smoothviz_dataloader(self):
        return [DataLoader(dataset_smoothviz, batch_size=self.training_conf.batch_size_train, shuffle=False, num_workers=0) for dataset_smoothviz in self.datasets_smoothviz]

    def predict_dataloader(
        self,
        date_list,
        species_list
    ):
        env_list = self.training_conf.env_list
        self._load_meta_list(
            date_list, 
            env_list, 
            species_list,
            'predict',
        )
        self.env_stack_predict = torch.load(os.path.join(self.tmp_path, f'env_stack_predict_{self.device}_{self.hostname}.pth'), map_location='cpu')
        self.embedding_predict = torch.load(os.path.join(self.tmp_path, f'embedding_predict_{self.device}_{self.hostname}.pth'), map_location='cpu')
        self.label_stack_predict = torch.load(os.path.join(self.tmp_path, f'label_stack_predict_{self.device}_{self.hostname}.pth'), map_location='cpu')
        self.k2_stack_predict = torch.load(os.path.join(self.tmp_path, f'k2_stack_predict_{self.device}_{self.hostname}.pth'), map_location='cpu')
        self.datasets_predict = []
#         print ("Setting up dataset for prediction...")
        for idx_species_date in range(len(self.label_stack_predict['species_date'])):

            self.datasets_predict.append(
                TaxaDataset_smoothviz(
                    idx_species_date, 
                    self.env_stack_predict, 
                    self.embedding_predict, 
                    self.label_stack_predict, 
                    self.training_conf.subsample_height, 
                    self.training_conf.subsample_width, 
                    self.training_conf.num_predict_steps, 
                    self.geo_extent,
                    self.device
                )
            )
        return [
            DataLoader(
                dataset_predict, 
                batch_size=self.training_conf.batch_size_predict, 
                shuffle=False, 
                num_workers=0
            ) 
            for dataset_predict in self.datasets_predict
        ]