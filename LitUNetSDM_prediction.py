import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from Unet import Unet
from types import SimpleNamespace
import os
import rasterio
import yaml
import shutil
import time
import logging
import h5py

class LitUNetSDM(pl.LightningModule):
    def __init__(self, 
                 custom_device, 
                 yaml_conf = './DeepSDM_conf.yaml', 
                 tmp_path = './tmp', 
                 predict_attention = False  # whether to plot attention score map while predicting results
                ):
        
        super().__init__()
            
        self.custom_device = custom_device
        self.tmp_path = tmp_path
        self.yaml_conf = yaml_conf

        with open(yaml_conf, 'r') as f:
            DeepSDM_conf = SimpleNamespace(**yaml.load(f, Loader = yaml.FullLoader))
        self.DeepSDM_conf = DeepSDM_conf
        self.training_conf = SimpleNamespace(**DeepSDM_conf.training_conf)
        
        self.model = Unet(num_vector = DeepSDM_conf.embedding_conf['num_vector'],
                          num_env = len(self.training_conf.env_list), 
                          height = self.training_conf.subsample_height, 
                          width = self.training_conf.subsample_width,
                        ).to(self.custom_device)
        
        self.no_data = -9999
        self.predict_attention = predict_attention
    

    def predict(self, dataloaders_predict=[], datamodule=None, output_dir='./predicts', ref_geotiff=os.path.join('./workspace', 'extent_binary.tif')):
        def setup_output_dirs(base_dir, meta_files, subdirs):
            os.makedirs(base_dir, exist_ok=True)
            with open(os.path.join(base_dir, 'DeepSDM_conf.yaml'), 'w') as f:
                yaml.dump(vars(self.DeepSDM_conf), f)
            for key, path in meta_files.items():
                shutil.copy(path, os.path.join(base_dir, f'{key}.json'))
            return {subdir: os.makedirs(os.path.join(base_dir, subdir), exist_ok=True) or os.path.join(base_dir, subdir) for subdir in subdirs}
        
        def create_species_subdirs(dirs, species_list):
            for d in dirs.values():
                for species in species_list:
                    os.makedirs(os.path.join(d, species), exist_ok = True)
        
        def load_geotiff(filepath):
            with rasterio.open(filepath) as ref:
                return ref.crs, ref.read(1), ref.transform

        def initialize_tensors(dataloader, device):
            height, width = dataloader.dataset.height_new, dataloader.dataset.width_new
            return (torch.zeros(height, width, device=device),
                    torch.zeros(height, width, device=device, dtype=torch.int32),
                    [] if self.predict_attention else None)

        def process_batch(inputs, embeddings, h_start, h_end, w_start, w_end, result, counts, attention):
            outputs, A = self.model(inputs.to(self.custom_device), embeddings.to(self.custom_device))
            outputs = torch.sigmoid(outputs).to(self.custom_device)
            for i in range(len(h_start)):
                result[h_start[i]:h_end[i], w_start[i]:w_end[i]] += outputs[i, 0, :, :]
                counts[h_start[i]:h_end[i], w_start[i]:w_end[i]] += 1
            if self.predict_attention:
                attention.append(
                    torch.cat(
                        [A, 
                         h_start[:, None].to(self.custom_device), 
                         w_start[:, None].to(self.custom_device), 
                         h_end[:, None].to(self.custom_device), 
                         w_end[:, None].to(self.custom_device)], 
                        dim=1)
                )
            return attention

        def normalize_result(result, counts, extent, nan_tensor, height_original, width_original, subsample_h, subsample_w):
            counts = torch.where(counts == 0, 1, counts).to(self.custom_device)
            normalized = (result / counts).to(self.custom_device)
            cropped = normalized[subsample_h:subsample_h + height_original, subsample_w:subsample_w + width_original].to(self.custom_device)
            return cropped.where(extent == 1, nan_tensor).to(self.custom_device)

        def save_results(result, label, species_date, dirs, extent_binary, crs, transform):
            result_np = result.cpu().numpy()
            sp_date = species_date[0]
            sp, date = '_'.join(sp_date.split('_')[:-1]), sp_date.split('_')[-1]

            plt.imshow(result_np, cmap='coolwarm', vmin=0, vmax=1)
            plt.colorbar()
            plt.plot(np.where(label == 1)[1], np.where(label == 1)[0], '.', color='black', markersize=1)
            plt.title(f'{sp}: {date}')
            plt.savefig(os.path.join(dirs['png'], sp, f'{sp}_{date}_predict.png'), dpi=200)
            plt.close()

            h5_file_path = os.path.join(dirs['h5'], sp, f'{sp}.h5')
        
            with h5py.File(h5_file_path, 'a') as hf:
                hf.attrs['crs'] = str(crs)
                hf.attrs['transform'] = transform.to_gdal()
                hf.attrs['nodata'] = self.no_data

                if date in hf:
                    del hf[date]
                hf.create_dataset(date, data = result_np * extent_binary, compression = 'gzip')
        
            return {f'{sp}_{date}': np.stack([label, result_np])}

        def save_attention_maps(dataloader, attention, dirs, species_date, extent_binary, crs, transform, height_original, width_original, subsample_h, subsample_w):
            sp_date = species_date[0]
            sp, date = '_'.join(sp_date.split('_')[:-1]), sp_date.split('_')[-1]

            height_start = attention[:, -4].to(torch.int16)
            width_start = attention[:, -3].to(torch.int16)
            height_end = attention[:, -2].to(torch.int16)
            width_end = attention[:, -1].to(torch.int16)

            hdf5_filepath = os.path.join(dirs['attention'], sp, f'{sp}_{date}_attention.h5')

            # attention map initialization
            height, width = dataloader.dataset.height_new, dataloader.dataset.width_new

            with h5py.File(hdf5_filepath, 'w') as hf:
                hf.attrs['crs'] = str(crs)
                hf.attrs['transform'] = transform.to_gdal()
                hf.attrs['nodata'] = self.no_data
                
                for i_env, env in enumerate(self.training_conf.env_list):
                    attention_map = torch.zeros(height, width, dtype=torch.float32, device=self.custom_device)
                    counts_map = torch.zeros(height, width, dtype=torch.int16, device=self.custom_device)

                    for i in range(len(attention)):
                        attention_map[height_start[i]:height_end[i], width_start[i]:width_end[i]] += attention[i, i_env]
                        counts_map[height_start[i]:height_end[i], width_start[i]:width_end[i]] += 1

                    counts_map = torch.where(counts_map == 0, 1, counts_map).to(self.custom_device)
                    attention_map /= counts_map
                    attention_map = attention_map[subsample_h:subsample_h + height_original, subsample_w:subsample_w + width_original].detach().cpu().numpy()
                    attention_map = np.where(extent_binary == 1, attention_map, self.no_data)

                    hf.create_dataset(env, data=attention_map, compression='gzip')                    
                    

        subdirs = ['h5', 'png'] + (['attention'] if self.predict_attention else [])
        dirs = setup_output_dirs(output_dir, self.DeepSDM_conf.meta_json_files, subdirs)
        create_species_subdirs(dirs, self.DeepSDM_conf.training_conf['species_list_predict'])
        extent_crs, extent_binary, extent_transform = load_geotiff(ref_geotiff)

        nan_tensor = torch.tensor(float('nan'), device=self.custom_device)
        extent = datamodule.geo_extent[0].to(self.custom_device)
        subsample_h, subsample_w = self.training_conf.subsample_height, self.training_conf.subsample_width
        height_original, width_original = datamodule.label_stack_predict['tensor'].shape[1:]
        results = {}
            
        logging.basicConfig(
            filename='logfile.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )            
    
        for loader in dataloaders_predict:
            loader.dataset.async_cuda()
            result, counts, attention = initialize_tensors(loader, self.custom_device)
            start_time = time.time()
            for inputs, embeddings, (h_start, h_end, w_start, w_end), species_date in loader:
                attention = process_batch(inputs, embeddings, h_start, h_end, w_start, w_end, result, counts, attention)
            
            normalized_result = normalize_result(result, counts, extent, nan_tensor, height_original, width_original, subsample_h, subsample_w)
            label = loader.dataset.label.numpy()
            results.update(save_results(normalized_result, label, species_date, dirs, extent_binary, extent_crs, extent_transform))

            if self.predict_attention and attention:
                attention = torch.cat(attention, dim=0)
                save_attention_maps(loader, attention, dirs, species_date, extent_binary, extent_crs, extent_transform,
                                    height_original, width_original, subsample_h, subsample_w)
            logging.info(f'{species_date[-1]}: {(time.time() - start_time):.2f} seconds. ({self.custom_device})')
        return results