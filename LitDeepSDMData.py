import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import rasterio
from TaxaDataset_smoothviz import TaxaDataset_smoothviz
from TaxaDataset import TaxaDataset
import time
import pytorch_lightning as pl
import yaml
from types import SimpleNamespace
import h5py

class LitDeepSDMData(pl.LightningDataModule):
    def __init__(self, 
                 yaml_conf = './DeepSDM_conf.yaml', 
                 tmp_path = './tmp'):
        
        super().__init__()
        
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
            print(item_[1])
            with open(item_[1]) as f:
                setattr(self, item_[0], json.load(f))

        with rasterio.open(geo_extent_file) as f:
            self.geo_extent = ToTensor()(f.read(1))
            self.geo_transform = f.transform
            self.geo_crs = f.crs
        
        self.DeepSDM_conf = DeepSDM_conf
        self.training_conf = SimpleNamespace(**DeepSDM_conf.training_conf)
        
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
                    img_ = ToTensor()(f.read(1)).cuda()
                img = img_.where(self.geo_extent.cuda() == 1, torch.normal(self.env_inf['info'][env_]['mean'], self.env_inf['info'][env_]['sd'], img_.shape).cuda())
                
                # environment factors which should be normalized
                if env_ not in self.training_conf.non_normalize_env_list:
                    img_norm = (img - self.env_inf['info'][env_]['mean']) / self.env_inf['info'][env_]['sd']
                else:
                    img_norm = img
                date_env_list.append(img_norm)
            env_tensor_list.append(torch.cat(date_env_list)[None, ])
        env_stack['tensor'] = torch.cat(env_tensor_list)  # env_stack['tensor'].shape = (len(stage_date_list), len(env_list), height, width)

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
            for date_ in stage_date_list:
                label_stack['species_date'].append(f'{species_}_{date_}')
                label_stack['species'].append(f'{species_}')
                label_stack['date'].append(f'{date_}')
                if species_ in self.sp_inf['file_name']:
                    with rasterio.open(os.path.join(self.sp_inf['dir_base'], f"{self.sp_inf['file_name'][species_][date_]}")) as f:
                        label_tensor_list.append(ToTensor()(f.read(1)))
                else:
                    label_tensor_list.append(torch.zeros([1, 
                                                          self.DeepSDM_conf.spatial_conf_tmp['grid_size'] * self.DeepSDM_conf.spatial_conf_tmp['num_of_grid_y'], 
                                                          self.DeepSDM_conf.spatial_conf_tmp['grid_size'] * self.DeepSDM_conf.spatial_conf_tmp['num_of_grid_x']]))
        t = torch.cat(label_tensor_list)
        label_stack['tensor'] = torch.where(t < 0, 0, t)
        
        return label_stack

    def _load_k2_list(self, stage_date_list, stage_env_list, stage_species_list):        
        # k2 value
        k2_stack = {
            'date' : stage_date_list,
        }
        k2_tensor_list = []
        for date_ in stage_date_list:
            with rasterio.open(os.path.join(self.k_inf['dir_base'], f"{self.k_inf['file_name'][date_]}")) as f:
                k2_tensor_list.append(ToTensor()(f.read(1)))
        k2_stack['tensor'] = torch.cat(k2_tensor_list)
        
        return k2_stack

    def _load_embedding_list(self, stage_date_list, stage_env_list, stage_species_list):        
        # embeddings
        embedding = dict()
        for species_ in stage_species_list:
            embedding[species_] = self.co_vec[species_]
            
        return embedding
            
    def _load_meta_list(self, stage_date_list, stage_env_list, stage_species_list, stage):
        torch.save(self._load_env_list(stage_date_list, stage_env_list, stage_species_list), f'{self.tmp_path}/env_stack_{stage}.pth')
        torch.save(self._load_embedding_list(stage_date_list, stage_env_list, stage_species_list), f'{self.tmp_path}/embedding_{stage}.pth')
        torch.save(self._load_label_list(stage_date_list, stage_env_list, stage_species_list), f'{self.tmp_path}/label_stack_{stage}.pth')
        torch.save(self._load_k2_list(stage_date_list, stage_env_list, stage_species_list), f'{self.tmp_path}/k2_stack_{stage}.pth')

    def prepare_data(self):
        
        torch.cuda.synchronize()
        start_time = time.time()

        file_missing = False
        for prefix_ in ['env_stack', 'embedding', 'label_stack', 'k2_stack']:
            for stage_ in ['train', 'val', 'smoothviz']:
                if not os.path.exists(f'{self.tmp_path}/{prefix_}_{stage_}.pth'):
                    file_missing = True
                if file_missing:
                    break
            if file_missing:
                break

        if file_missing:
            print("Missing some cached files, re-caching...")
            env_list = self.training_conf.env_list
            
            date_list_train = self.training_conf.date_list_train
            species_list_train = self.training_conf.species_list_train

            date_list_val = self.training_conf.date_list_val
            species_list_val = self.training_conf.species_list_val

            date_list_smoothviz = self.training_conf.date_list_smoothviz
            species_list_smoothviz = self.training_conf.species_list_smoothviz

            ### caching staged lists
            # train
            print("train")
            self._load_meta_list(
                date_list_train, 
                env_list, 
                species_list_train,
                'train'
            )

            # val
            print("val")
            self._load_meta_list(
                date_list_val, 
                env_list, 
                species_list_val,
                'val'
            )

            # smoothviz
            print("smoothviz")
            self._load_meta_list(
                date_list_smoothviz, 
                env_list, 
                species_list_smoothviz,
                'smoothviz'
            )

        torch.cuda.synchronize()
        print(f'Data prepared in {time.time() - start_time} seconds.')


    def setup(self, stage=None):

        ###
        torch.cuda.synchronize()
        start_time = time.time()
        self.env_stack_train = torch.load(f'{self.tmp_path}/env_stack_train.pth', map_location='cpu')
        self.embedding_train = torch.load(f'{self.tmp_path}/embedding_train.pth', map_location='cpu')
        self.label_stack_train = torch.load(f'{self.tmp_path}/label_stack_train.pth', map_location='cpu')
        self.k2_stack_train = torch.load(f'{self.tmp_path}/k2_stack_train.pth', map_location='cpu')
        print(f'train ############################################## {self.trainer.global_rank}')
        if self.trainer.global_rank == 0:
            dataset_train = TaxaDataset(
                self.env_stack_train, 
                self.embedding_train, 
                self.label_stack_train, 
                self.k2_stack_train, 'train', self.DeepSDM_conf, self.trainer.global_rank
            )
        else:
            dataset_train = None
        self.trainer.strategy.barrier()
        dataset_train = self.trainer.strategy.broadcast(dataset_train, src=0)
        self.dataset_train = dataset_train
        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data train loaded in {time.time() - start_time} seconds.')

        ###
        torch.cuda.synchronize()
        start_time = time.time()
        self.label_stack_val = torch.load(f'{self.tmp_path}/label_stack_val.pth', map_location='cpu')
        print(f'val ############################################## {self.trainer.global_rank}')
        if self.trainer.global_rank == 0:
            dataset_val = TaxaDataset(
                torch.load(f'{self.tmp_path}/env_stack_val.pth', map_location='cpu'), 
                torch.load(f'{self.tmp_path}/embedding_val.pth', map_location='cpu'), 
                self.label_stack_val, 
                torch.load(f'{self.tmp_path}/k2_stack_val.pth', map_location='cpu'), 'val', self.DeepSDM_conf, self.trainer.global_rank
            )
        else:
            dataset_val = None
        dataset_val = self.trainer.strategy.broadcast(dataset_val, src=0)
        self.dataset_val = dataset_val
        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data val loaded in {time.time() - start_time} seconds.')

        ###
        torch.cuda.synchronize()
        start_time = time.time()
        print(f'train on val ############################################## {self.trainer.global_rank}')
        if self.trainer.global_rank == 0:
            dataset_train_on_val = TaxaDataset(
                self.env_stack_train, 
                self.embedding_train, 
                self.label_stack_train, 
                self.k2_stack_train, 'val', self.DeepSDM_conf, self.trainer.global_rank
            )
        else:
            dataset_train_on_val = None
        dataset_train_on_val = self.trainer.strategy.broadcast(dataset_train_on_val, src=0)
        self.dataset_train_on_val = dataset_train_on_val
        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data train_on_val loaded in {time.time() - start_time} seconds.')
        
        ###
        ##########################################################        
        torch.cuda.synchronize()
        start_time = time.time()
        self.env_stack_smoothviz = torch.load(f'{self.tmp_path}/env_stack_smoothviz.pth', map_location='cpu')
        self.embedding_smoothviz = torch.load(f'{self.tmp_path}/embedding_smoothviz.pth', map_location='cpu')
        self.label_stack_smoothviz = torch.load(f'{self.tmp_path}/label_stack_smoothviz.pth', map_location='cpu')
        self.k2_stack_smoothviz = torch.load(f'{self.tmp_path}/k2_stack_smoothviz.pth', map_location='cpu')
        
        if self.trainer.global_rank == 0:
            self.datasets_smoothviz = []
            print ("Setting up dataset in log_img...")
            for idx_species_date in range(len(self.label_stack_smoothviz['species_date'])):

                self.datasets_smoothviz.append(
                    TaxaDataset_smoothviz(
                        idx_species_date, 
                        self.env_stack_smoothviz, self.embedding_smoothviz, self.label_stack_smoothviz, 
                        self.training_conf.subsample_height, self.training_conf.subsample_width, self.training_conf.num_smoothviz_steps
                    )
                )

        self.trainer.strategy.barrier()

        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data smoothviz loaded in {time.time() - start_time} seconds.')

        ##########################################################

#         if self.trainer.global_rank == 0:
#             self._save_trainval_split(self.dataset_train)

        self.trainer.strategy.barrier()
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, self.training_conf.batch_size_train, shuffle=True, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return [
            DataLoader(self.dataset_train_on_val, self.training_conf.batch_size_train, shuffle=False, num_workers=16, pin_memory=True), # eval train dataset first
            DataLoader(self.dataset_val, self.training_conf.batch_size_train, shuffle=False, num_workers=16, pin_memory=True), # Why shuffling here?
        ]

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
        self.env_stack_predict = torch.load(f'{self.tmp_path}/env_stack_predict.pth', map_location='cpu')
        self.embedding_predict = torch.load(f'{self.tmp_path}/embedding_predict.pth', map_location='cpu')
        self.label_stack_predict = torch.load(f'{self.tmp_path}/label_stack_predict.pth', map_location='cpu')
        self.k2_stack_predict = torch.load(f'{self.tmp_path}/k2_stack_predict.pth', map_location='cpu')
        self.datasets_predict = []
        print ("Setting up dataset for prediction...")

        for idx_species_date in range(len(self.label_stack_predict['species_date'])):

            self.datasets_predict.append(
                TaxaDataset_smoothviz(
                    idx_species_date, 
                    self.env_stack_predict, self.embedding_predict, self.label_stack_predict, 
                    self.training_conf.subsample_height, self.training_conf.subsample_width, self.training_conf.num_predict_steps
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