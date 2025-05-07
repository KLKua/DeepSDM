import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pytorch_lightning as pl
from Unet import Unet
from types import SimpleNamespace
from torchmetrics.functional.classification import binary_f1_score, binary_auroc
import mlflow
import os
import rasterio
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil


class LitUNetSDM(pl.LightningModule):
    def __init__(self, 
                 yaml_conf = './DeepSDM_conf.yaml', 
                 tmp_path = './tmp', 
                 predict_attention = False  # whether to plot attention score map while predicting results
                ):
        
        super().__init__()
        
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
                        )

        self._init_val_step_vars()
        self._init_train_step_vars()
        
        self.no_data = -9999
        self.predict_attention = predict_attention
        
    def forward(self, env_image, bio_vector):
        return self.model(env_image, bio_vector)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.training_conf.learning_rate)
        return optimizer
    
    def flatten_list(self, l):
        return [item for sublist in l for item in sublist]
    
    def _init_val_step_vars(self):
        self.val_step_outputs = []
        self.val_step_species = []
        self.val_step_date = []
        self.val_step_labels = []
        self.val_step_k2 = []

    def _init_train_step_vars(self):
        self.train_step_outputs = []
        self.train_step_species = []
        self.train_step_date = []
        self.train_step_labels = []
        self.train_step_k2 = []        
        
    def _step(self, batch, batch_idx, val=False, dataloader_idx=0):
        [inputs, embeddings], labels, k2, species, date = batch
        inputs = inputs.reshape(-1, *inputs.shape[-3:])
        embeddings = embeddings.reshape(-1, *embeddings.shape[-3:])
        labels = labels.reshape(-1, *labels.shape[-3:])
        k2 = k2.reshape(-1, *k2.shape[-3:])

        image_output, A = self.model(inputs, embeddings)
    
        ################## effort weighted bce loss
        
        zero_tensor = torch.tensor(0, device=self.device)
        
        k_matrix = torch.pow(k2, self.training_conf.k2_p)
        
        #l = self.bce_loss(image_output, labels)
        l = F.binary_cross_entropy_with_logits(image_output, labels.to(torch.float), reduction='none')
        l = l.where(k2 >= 0, zero_tensor) #torch.zeros(l.shape, device=l.device))
        
        #l1
        l1_matrix = l.where(labels == 1, zero_tensor) #torch.zeros(l.shape, device=l.device))
        l1_loss = l1_matrix.sum(axis = (1, 2, 3))
        l1_count = (labels == 1).sum(axis = (1, 2, 3))
        l1_loss = (l1_loss / l1_count).nan_to_num()
                
        #l2    
        l2_matrix = (k_matrix * l).where(((labels == 0) & (k2 > 0)), zero_tensor) #torch.zeros(l.shape, device=l.device))
        l2_loss = l2_matrix.sum(axis = (1, 2, 3))
        l2_count = ((labels == 0) & (k2 > 0)).sum(axis = (1, 2, 3))
        l2_loss = (l2_loss / l2_count).nan_to_num()


        #l3    
        l3_matrix = l.where(((labels == 0) & (k2 == 0)), zero_tensor) #torch.zeros(l.shape, device=l.device))
        l3_loss = l3_matrix.sum(axis = (1, 2, 3))
        l3_count = ((labels == 0) & (k2 == 0)).sum(axis = (1, 2, 3))
        l3_loss = (l3_loss / l3_count).nan_to_num() 

        k1_loss = l1_loss.sum() / len(image_output)
        k2_loss = self.training_conf.k2 * l2_loss.sum() / len(image_output)
        k3_loss = self.training_conf.k3 * l3_loss.sum() / len(image_output)
        total_loss = k1_loss + k2_loss + k3_loss

        metrics = dict(
            loss = total_loss,
            k1_loss = k1_loss,
            k2_loss = k2_loss,
            k3_loss = k3_loss,
        )

        if val is True:
            if dataloader_idx == 1: # val
                self.val_step_outputs.append(torch.sigmoid(image_output))
                self.val_step_species.append(species)
                self.val_step_date.append(date)
                self.val_step_labels.append(labels)
                self.val_step_k2.append(k2)
            elif dataloader_idx == 0: # train
                self.train_step_outputs.append(torch.sigmoid(image_output))
                self.train_step_species.append(species)
                self.train_step_date.append(date)
                self.train_step_labels.append(labels)
                self.train_step_k2.append(k2)
                
        else:
            pass
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        metrics = self._step(batch, batch_idx)
        
        self.log("train_loss", metrics['loss'], on_epoch=True, prog_bar=True, logger=True, on_step=False, sync_dist=True)
        self.log("train_k1_loss", metrics['k1_loss'], on_epoch=True, logger=True, on_step=False, sync_dist=True)
        self.log("train_k2_loss", metrics['k2_loss'], on_epoch=True, logger=True, on_step=False, sync_dist=True)
        self.log("train_k3_loss", metrics['k3_loss'], on_epoch=True, logger=True, on_step=False, sync_dist=True)
        return metrics['loss']

    
    def aggregate_loop_results_(self, result, labels, k2, species, date, dataset='val'):
        # ALL following flows should be ran under eval+no_grad mode
        
        if dataset == 'val':
            result_epoch = torch.cat(self.val_step_outputs, axis = 0)
            labels_epoch = torch.cat(self.val_step_labels, axis = 0)
            k2_epoch = torch.cat(self.val_step_k2, axis = 0)
            species_epoch = self.flatten_list(self.val_step_species)
            date_epoch = self.flatten_list(self.val_step_date)
            self._init_val_step_vars()
        elif dataset == 'train':
            result_epoch = torch.cat(self.train_step_outputs, axis = 0)
            labels_epoch = torch.cat(self.train_step_labels, axis = 0)
            k2_epoch = torch.cat(self.train_step_k2, axis = 0)
            species_epoch = self.flatten_list(self.train_step_species)
            date_epoch = self.flatten_list(self.train_step_date)
            self._init_train_step_vars()

        label_stack_val = self.trainer.datamodule.label_stack_val
        
        agg = dict(
            nop = [],
            pred_a_sample = [],
            pred_p_sample = []
        )
        
        agg = SimpleNamespace(**agg)

        num_of_species_dates = len(label_stack_val['species_date'])
        for idx_species_date in range(num_of_species_dates):
            species_date = label_stack_val['species_date'][idx_species_date]
            species = label_stack_val['species'][idx_species_date]
            date = label_stack_val['date'][idx_species_date]

            # Find the row idx of occurrences for specific species and date
            idx_epoch = [i for i in range(len(species_epoch)) if (species_epoch[i] == species) & (date_epoch[i] == date)]

            # The predicted p according to the found rows
            pred_epoch = result_epoch[idx_epoch, :, :].view(-1)#.detach()

            # The true values according to the found rows
            true_epoch = labels_epoch[idx_epoch, :, :].view(-1)

            # The k2 values according to the found rows
            k2_use_epoch = k2_epoch[idx_epoch, :, :].view(-1)
            
            # ignoring calculation if no occurrence found
#             if (sum(true_epoch == 1) == 0):
#                 continue
            
            true_epoch_eq1 = true_epoch == 1
    
            # number of occurrence points
            nop_epoch = true_epoch_eq1.sum() #sum(true_epoch == 1)
            # where no occurrence but WHAT IS THIS k2_use_val <= 1? Surveyed but no occurrence?
            
            # for thoese k2_use_epoch out of extent are set to -9999
            pred_epoch_a_all = pred_epoch[torch.where((~true_epoch_eq1) & (k2_use_epoch >= 0))[0]]


            # draw random points from surveyed but no occurrence
#             pred_epoch_a_sample = random.sample(pred_epoch_a_all, nop_epoch)

            indice = torch.tensor(random.sample(range(pred_epoch_a_all.shape[0]), nop_epoch), dtype=int)
            pred_epoch_a_sample = pred_epoch_a_all[indice]

        
            # predicted p on those points of occurrence
            pred_epoch_p_sample = pred_epoch[true_epoch_eq1] #.detach() #.cpu().tolist()

#             print(pred_epoch_a_all.shape, pred_epoch_a_sample.shape, pred_epoch_p_sample.shape)


            agg.nop.append(nop_epoch)
#             agg.pred_a_all.append(pred_epoch_a_all)
            agg.pred_a_sample.append(pred_epoch_a_sample)
            agg.pred_p_sample.append(pred_epoch_p_sample)

#         torch.cuda.synchronize()
#         print(f'Agg {dataset} results takes {time.time() - start_time} seconds.')

        return agg
        
    def on_validation_epoch_end(self):
#         with torch.no_grad():
        if (self.trainer.global_rank == 0) and (self.current_epoch % 5 == 0):
            torch.cuda.synchronize()
            start_time = time.time()            
            self.log_img()
            torch.cuda.synchronize()
            print(f'log_img: {time.time() - start_time} seconds ..................')


#         print(f"{self.trainer.global_rank} is blocked here.")
        self.trainer.strategy.barrier()
#         print(f"{self.trainer.global_rank} is resumed here.")

        agg_train = self.aggregate_loop_results_(
            self.train_step_outputs, self.train_step_labels, self.train_step_k2, self.train_step_species, self.train_step_date, dataset='train')

        agg_val = self.aggregate_loop_results_(
            self.val_step_outputs, self.val_step_labels, self.val_step_k2, self.val_step_species, self.val_step_date, dataset='val')


#         threshold_options = np.arange(1e-3, 1, 1e-3)
        threshold_options = np.concatenate([
            np.arange(1e-3, 5e-2, 2.5e-3),
            np.arange(5e-2, 1e-1, 5e-3),
            np.arange(1e-1, 2.5e-1, 1e-2),
            np.arange(2.5e-1, 5e-1, 2.5e-2),
            np.arange(5e-1, 1, 5e-2)
        ])
#         print(threshold_options.shape)

        f1_epoch_train = []
        best_thresholds = []
        f1_epoch_val = []
        num_presence_epoch = []

        for i_ in range(len(agg_train.nop)):
            nop_train = agg_train.nop[i_]
            nop_val = agg_val.nop[i_]

            if nop_train == 0 or nop_val == 0:
                continue

            pred_train_a_sample = agg_train.pred_a_sample[i_]
            pred_val_a_sample = agg_val.pred_a_sample[i_]

            pred_train_p_sample = agg_train.pred_p_sample[i_]
            pred_val_p_sample = agg_val.pred_p_sample[i_]

            f1 = []
            for threshold in threshold_options:

                f1.append(binary_f1_score(
                    torch.cat([pred_train_p_sample, pred_train_a_sample]),
                    torch.tensor([1] * nop_train + [0] * nop_train, device=self.device), threshold=threshold))

            threshold_best = threshold_options[torch.tensor(f1).argmax()]
            best_thresholds.append(threshold_best)
            f1_best_train = max(f1)
#             f1_best_val = f1_score(torch.tensor([1] * nop_val + [0] * nop_val), torch.where(torch.tensor(pred_val_p_sample + pred_val_a_sample) >= threshold_best, 1, 0), zero_division = 1)
            f1_best_val = binary_f1_score(
                torch.cat([pred_val_p_sample, pred_val_a_sample]), 
                torch.tensor([1] * nop_val + [0] * nop_val, device=self.device), threshold=threshold_best)

            f1_epoch_train.append(f1_best_train)
            f1_epoch_val.append(f1_best_val)
#             num_presence_epoch.append(nop_val.cpu().tolist())
            num_presence_epoch.append(nop_val)

        len_f1_epoch_train = len(f1_epoch_train)
        len_f1_epoch_val = len(f1_epoch_val)
        assert(len_f1_epoch_train == len_f1_epoch_train)
        sum_num_presence_epoch = sum(num_presence_epoch)

        f1_epoch_train = torch.tensor(f1_epoch_train, device=self.device)
        f1_epoch_val = torch.tensor(f1_epoch_val, device=self.device)
        num_presence_epoch = torch.tensor(num_presence_epoch, device=self.device)
        best_thresholds = torch.tensor(best_thresholds, device=self.device)

        if len_f1_epoch_train > 0:
            f1_train_avg = torch.mean(f1_epoch_train) # sum(f1_epoch_train) / len_f1_epoch_train
            f1_train_weighted_avg = torch.sum(f1_epoch_train * num_presence_epoch) / sum_num_presence_epoch
            best_threshold_avg = torch.mean(best_thresholds)
            # sum([f1_epoch_train[i] * num_presence_epoch[i] / sum_num_presence_epoch for i in range(len_f1_epoch_train)])
        else:
            f1_train_avg = 0
            f1_train_weighted_avg = 0
            best_threshold_avg = 0

        if len_f1_epoch_val > 0:
            f1_val_avg = torch.mean(f1_epoch_val) # sum(f1_epoch_train) / len_f1_epoch_train
            f1_val_weighted_avg = torch.sum(f1_epoch_val * num_presence_epoch) / sum_num_presence_epoch
        else:
            f1_val_avg = 0
            f1_val_weighted_avg = 0

        if self.trainer.state.stage != 'sanity_check':
            self.log('f1_val', f1_val_avg, on_epoch=True, on_step=False, sync_dist=True)
            self.log('f1_val_weighted', f1_val_weighted_avg, on_epoch=True, on_step=False, sync_dist=True)
            self.log('f1_train', f1_train_avg, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
            self.log('f1_train_weighted', f1_train_weighted_avg, on_epoch=True, on_step=False, sync_dist=True)
            self.log('best_threshold_avg', best_threshold_avg, on_epoch=True, on_step=False, sync_dist=True)

        torch.cuda.empty_cache()

        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        metrics = self._step(batch, batch_idx, val=True, dataloader_idx=dataloader_idx)
        
        self.log("val_loss", metrics['loss'], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_k1_loss", metrics['k1_loss'], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_k2_loss", metrics['k2_loss'], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_k3_loss", metrics['k3_loss'], on_epoch=True, on_step=False, sync_dist=True)
        return metrics['loss']
    
    
    ####################################################################################################################
    ##
    ##  TODO: THIS PART SHOULD BE SOMEHOW REFACTORED TO LIGHTNING CONVENTION
    ##
    ####################################################################################################################
    
    def log_img(self):
#         return
#         print(f'################### {self.trainer.state.stage}')
        nan_tensor = torch.tensor(float('nan'), device=self.device)
        
        extent = self.trainer.datamodule.geo_extent[0].to(self.device) #1是預測範圍;0是非預測範圍
        label_stack_smoothviz = self.trainer.datamodule.label_stack_smoothviz
        env_stack_smoothviz = self.trainer.datamodule.env_stack_smoothviz
        embedding_smoothviz = self.trainer.datamodule.embedding_smoothviz

        height_original = label_stack_smoothviz['tensor'].shape[1]
        width_original = label_stack_smoothviz['tensor'].shape[2]
        
        subsample_height = self.training_conf.subsample_height
        subsample_width = self.training_conf.subsample_width
         
        dataset_train = self.trainer.datamodule.train_dataloader().dataset
        partition = dataset_train.split_tif[:height_original, :width_original].to(self.device) #1是train部分;0是非train部分(含陸地和海)

        roc_epoch_val = []
        roc_epoch_train = []

        num_of_dates = len(self.training_conf.date_list_smoothviz)
#         print(label_stack_smoothviz['species_date'])
        ncols = min(num_of_dates, 4)
        nrows = int(np.ceil(num_of_dates // ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, max(8,2*ncols*nrows)))
        # REFACTOR THE DATALOADING PROCEDURE WOULD HELP

        dataloaders_smoothviz = self.trainer.datamodule.smoothviz_dataloader()
        for _ in dataloaders_smoothviz:
            _.dataset.async_cuda()

        print("\n")
        for dataloader_idx, dataloader_smoothviz in enumerate(dataloaders_smoothviz):

#             dataloader_smoothviz = DataLoader(dataset_smoothviz, batch_size=self.conf.batch_size * self.conf.num_train_subsample_stacks, shuffle=False, num_workers=16, pin_memory=True)

            height_new = dataloader_smoothviz.dataset.height_new
            width_new = dataloader_smoothviz.dataset.width_new

            result = torch.zeros(height_new, width_new, device=self.device) # 放大後的尺寸
            counts = torch.zeros(height_new, width_new, device=self.device, dtype=torch.int32)

            torch.cuda.synchronize()
            start_time = time.time()            
            for inputs, embeddings, (height_start, height_end, width_start, width_end), species_date in dataloader_smoothviz:
                outputs, A = self.model(inputs.to(self.device), embeddings.to(self.device))
                outputs = torch.sigmoid(outputs)
                for i in range(len(species_date)):
                    result[height_start[i]:height_end[i], width_start[i]:width_end[i]] += outputs[i, 0, :, :]
                    counts[height_start[i]:height_end[i], width_start[i]:width_end[i]] += 1
            
            dataloader_smoothviz.dataset.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f'{species_date[i]}: {time.time() - start_time} seconds ..................', end='\r')

            counts = torch.where(counts == 0, 1, counts)
            result = result / counts
            result = result[subsample_height: (subsample_height + height_original),
                            subsample_width: (subsample_width + width_original)]

            #算rocauc
            #製作pseudo-absence
            label = dataloader_smoothviz.dataset.label.to(self.device) #.detach().to(dev)
            
            ### REWRITE THIS PART
            #y, x = torch.where((extent == 1) & (label == 0))[0], torch.where((extent == 1) & (label == 0))[1] # val部份的所有點位（只有陸地）
            y, x = torch.where((extent == 1) & (label == 0))
            num_all = len(y)
#             random_num = np.random.choice(num_all, 10000, replace = False) # 隨機選val部份的所有點位（只有陸地）中的 10000個點位
            random_indice = torch.tensor(random.sample(range(num_all), 10000), dtype=int)
            bg = torch.zeros_like(partition, device=self.device) #這是pseudo-absence的圖片
            bg[y[random_indice], x[random_indice]] = 1 # bg這個影像中，如果是選到的pseudo-absence值是1; 其餘是0

#             try:

            label_one_trian_ = result[(label == 1) & (partition == 1)]
#             assert(len(label_one_trian_) > 0)
            if len(label_one_trian_) > 0:
                bg_one_train_ = result[(bg == 1) & (partition == 1)]
                train_pred = torch.cat((bg_one_train_, label_one_trian_))
                train_true = torch.cat([torch.zeros_like(bg_one_train_, device=self.device), torch.ones_like(label_one_trian_, device=self.device)])
                roc_score_train = binary_auroc(train_pred, train_true.to(torch.int))
                roc_epoch_train.append(roc_score_train)
#             except ValueError:
#                 roc_score_train = -9

#             try:
            label_one_val_ = result[(label == 1) & (partition == 0)]
#             assert(len(label_one_val_) > 0)
            if len(label_one_val_) > 0:
                bg_one_val_ = result[(bg == 1) & (partition == 0)]
                val_pred = torch.cat((bg_one_val_, label_one_val_)) #.detach().cpu().numpy()
                val_true = torch.cat([torch.zeros_like(bg_one_val_, device=self.device), torch.ones_like(label_one_val_, device=self.device)])
                roc_score_val = binary_auroc(val_pred, val_true.to(torch.int))
                roc_epoch_val.append(roc_score_val)
#             except ValueError:
#                 roc_score_val = -9
        
            ### Plot map and occurrence points
            # start_time = time.time()
                        
            result_masked = result.where(extent == 1, nan_tensor)
            result_masked_npy = result_masked.cpu().numpy()

            try:
                if dataloader_idx % ncols == 0:
                    for ax in axes.ravel():
                        ax.clear()

                if nrows > 1:
                    ax = axes[int(dataloader_idx//ncols)][dataloader_idx%ncols]
                else:
                    ax = axes[dataloader_idx%ncols]
            except:
                ax = axes
            # if self.trainer.global_rank == 0:
            ax.imshow(result_masked_npy, cmap = 'coolwarm', vmin = 0, vmax = 1)

            # black points: presence points in training area
            # green points: presence points in validation area
            label_cpu = label.cpu().numpy()
            partition_cpu = partition.cpu().numpy()
            ax.plot(np.where((label_cpu == 1) & (partition_cpu == 1))[1], np.where((label_cpu == 1) & (partition_cpu == 1))[0], '.', color = 'black')
            ax.plot(np.where((label_cpu == 1) & (partition_cpu == 0))[1], np.where((label_cpu == 1) & (partition_cpu == 0))[0], '.', color = 'green')

            #title_ = f'{species_date[0]}_ep{self.current_epoch:04d}'


#             date_ = species_date[0].split('_')[-1]
            # 8/30 修改
            date_ = dataloader_smoothviz.dataset.date.split('_')[-1]
            
            ax.set_title(date_)
            
            plt.draw()
            plt.pause(0.001)
#             plt.axis('off')        
#             wandb.log({f"{species_time[0]}": [wandb.Image(plt, mode = 'L')]}, 
#                      step = epoch)  
#             plt.close('all')
#             taiwan_map = result_img

            if dataloader_idx % ncols == ncols - 1:
                sp_ = '_'.join(species_date[0].split('_')[:-1])
                fig.suptitle(f'{self.trainer.state.stage}: {sp_}_ep{self.current_epoch:04d}_smoothviz')
                plt.tight_layout()
#                 plt.savefig(f'tmp/{sp_}_ep{self.current_epoch:04d}_smoothviz.png')
                self.logger.experiment.log_figure(figure = fig, artifact_file = f'{sp_}_ep{self.current_epoch:04d}_smoothviz.png', run_id = self.logger.run_id)

#             print(time.time() - start_time)

            ax.clear()
        
        
        
        auc_val = sum(roc_epoch_val) / len(roc_epoch_val) #sum([i for i in roc_epoch_val if i > 0]) / len([i for i in roc_epoch_val if i > 0])
        auc_train = sum(roc_epoch_train) / len(roc_epoch_train) # sum([i for i in roc_epoch_train if i > 0]) / len([i for i in roc_epoch_train if i > 0])
        
        plt.close()
        
        if self.trainer.state.stage != 'sanity_check':
            self.log('auc_train', auc_train, on_epoch=True, on_step=False, rank_zero_only=True, sync_dist=False)
            self.log('auc_val', auc_val, on_epoch=True, on_step=False, rank_zero_only=True, sync_dist=False)

            
    def predict(self, dataloaders_predict=[], datamodule=None, output_dir='./predicts', ref_geotiff = './workspace/extent_binary.tif'):
        
        # log the yaml files to the predicts folder
        dir_out = f'{output_dir}'
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        with open(f'{output_dir}/DeepSDM_conf.yaml', 'w') as f:
            yaml.dump(vars(self.DeepSDM_conf), f)
        shutil.copy(self.DeepSDM_conf.meta_json_files['env_inf'], f'{output_dir}/env_information.json')
        shutil.copy(self.DeepSDM_conf.meta_json_files['sp_inf'], f'{output_dir}/sp_information.json')
        
        dir_tif_out = f'{output_dir}/tif'
        if not os.path.isdir(dir_tif_out):
            os.makedirs(dir_tif_out)

        dir_png_out = f'{output_dir}/png'
        if not os.path.isdir(dir_png_out):
            os.makedirs(dir_png_out)
            
        if self.predict_attention:
            dir_attention_out = f'{output_dir}/attention'
            if not os.path.isdir(dir_attention_out):
                os.makedirs(dir_attention_out)

        nan_tensor = torch.tensor(float('nan'), device=self.device)
        extent = datamodule.geo_extent[0].to(self.device) #1是預測範圍;0是非預測範圍
        subsample_height = self.training_conf.subsample_height
        subsample_width = self.training_conf.subsample_width
  
        label_stack_predict = datamodule.label_stack_predict
        env_stack_predict = datamodule.env_stack_predict
        embedding_predict = datamodule.embedding_predict

        height_original = label_stack_predict['tensor'].shape[1]
        width_original = label_stack_predict['tensor'].shape[2]

#         dataset_train = self.trainer.datamodule.train_dataloader().dataset
#         partition = dataset_train.split_tif[:height_original, :width_original].to(self.device) #1是train部分;0是非train部分(含陸地和海)

#         fig, ax = plt.subplots(nrows=1) #nrows, ncols=ncols, figsize=(12, 8))

        with rasterio.open(ref_geotiff) as ref:
            extent_crs = ref.crs
            extent_binary = ref.read(1)
            extent_transform = ref.transform

        for _ in dataloaders_predict:
            _.dataset.async_cuda()

        results = dict()
        for dataloader_idx, dataloader_predict in enumerate(dataloaders_predict):
            
            plt.figure()

            height_new = dataloader_predict.dataset.height_new
            width_new = dataloader_predict.dataset.width_new

            result = torch.zeros(height_new, width_new, device=self.device) # 放大後的尺寸
            counts = torch.zeros(height_new, width_new, device=self.device, dtype=torch.int32)
            
            # tensor to load attention score
            attention = None
            
            torch.cuda.synchronize()
            start_time = time.time()            
            for inputs, embeddings, (height_start, height_end, width_start, width_end), species_date in dataloader_predict:
                outputs, A = self.model(inputs.to(self.device), embeddings.to(self.device))
                outputs = torch.sigmoid(outputs)
                
                # concatenate attention scores
                if self.predict_attention:
                    if attention is None:
                        attention = torch.cat((A, 
                                               height_start[:, None].to(self.device), 
                                               width_start[:, None].to(self.device), 
                                               height_end[:, None].to(self.device), 
                                               width_end[:, None].to(self.device)), axis = 1)
                    else:
                        attention = torch.cat((attention, 
                                               torch.cat((A, 
                                                          height_start[:, None].to(self.device), 
                                                          width_start[:, None].to(self.device), 
                                                          height_end[:, None].to(self.device), 
                                                          width_end[:, None].to(self.device)), axis = 1)), axis = 0)
                
                for i in range(len(species_date)):
                    result[height_start[i]:height_end[i], width_start[i]:width_end[i]] += outputs[i, 0, :, :]
                    counts[height_start[i]:height_end[i], width_start[i]:width_end[i]] += 1
            
            dataloader_predict.dataset.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f'{species_date[i]}: {time.time() - start_time} seconds ..................', end='\r')

            counts = torch.where(counts == 0, 1, counts)
            result = result / counts
            result = result[subsample_height: (subsample_height + height_original),
                            subsample_width: (subsample_width + width_original)]

            #算rocauc
            #製作pseudo-absence
            label = dataloader_predict.dataset.label.numpy() #.detach().to(dev)

            
            result_masked = result.where(extent == 1, nan_tensor)
            result_masked_npy = result_masked.cpu().numpy()

            # if self.trainer.global_rank == 0:
            plt.imshow(result_masked_npy, cmap = 'coolwarm', vmin = 0, vmax = 1)
            plt.colorbar()

            # black points: presence points in training area
            # green points: presence points in validation area
            plt.plot(np.where(label == 1)[1], np.where(label == 1)[0], '.', color = 'black', markersize = 1)

            date_ = species_date[0].split('_')[-1:][0]
            sp_ = '_'.join(species_date[0].split('_')[:-1])
            
            
            # save the logged attention score as map
            if self.predict_attention:
                height_start = attention[:, -4].to(torch.int16)
                width_start = attention[:, -3].to(torch.int16)
                height_end = attention[:, -2].to(torch.int16)
                width_end = attention[:, -1].to(torch.int16)
                for i_env in range(len(self.training_conf.env_list)):
                    attention_map = torch.zeros(height_new, width_new, device = self.device) # 放大後的尺寸
                    counts_map = torch.zeros(height_new, width_new, device = self.device, dtype=torch.int16)
                    for i_attention in range(len(attention)):
                        attention_map[height_start[i_attention]:height_end[i_attention], width_start[i_attention]:width_end[i_attention]] = attention_map[height_start[i_attention]:height_end[i_attention], width_start[i_attention]:width_end[i_attention]] + attention[i_attention, i_env]
                        counts_map[height_start[i_attention]:height_end[i_attention], width_start[i_attention]:width_end[i_attention]] = counts_map[height_start[i_attention]:height_end[i_attention], width_start[i_attention]:width_end[i_attention]] + 1
#                 torch.save(attention, f"{dir_attention_out}/{sp_}_{date_}_attention.pt")
                    counts_map = torch.where(counts_map == 0, 1, counts_map)
                    attention_map = attention_map / counts_map
                    attention_map = attention_map[subsample_height:(subsample_height + height_original), 
                                                  subsample_width:(subsample_width + width_original)].detach().cpu().numpy()
                    attention_map = np.where(extent_binary == 1, attention_map, self.no_data)
                    with rasterio.open(
                        f"{dir_attention_out}/{sp_}_{date_}_attention_{self.training_conf.env_list[i_env]}.tif", 
                        'w', 
                        height = extent_binary.shape[0], 
                        width = extent_binary.shape[1],
                        count = 1, 
                        nodata = self.no_data, 
                        crs = extent_crs, 
                        dtype = rasterio.float32, 
                        transform = extent_transform
                    ) as dst:
                        dst.write(attention_map, 1)
                    
            plt.title(f'{sp_}: {date_}')
            plt.savefig(f'{dir_png_out}/{sp_}_{date_}_predict.png', dpi = 200)
            plt.close()
            
            with rasterio.open(
                f"{dir_tif_out}/{sp_}_{date_}_predict.tif", 
                'w', 
                height = extent_binary.shape[0], 
                width = extent_binary.shape[1],
                count = 1, 
                nodata = self.no_data, 
                crs = extent_crs, 
                dtype = rasterio.float32, 
                transform = extent_transform
            ) as dst:
                dst.write(result_masked_npy * extent_binary, 1)
            
            results[f'{sp_}_{date_}'] = np.stack([label, result_masked_npy])
            
        return results
    
    def on_fit_start(self):
        if self.trainer.global_rank == 0:
            self.tmp_path = f'{self.tmp_path}/{self.training_conf.experiment_name}'
            if not os.path.isdir(self.tmp_path):
                os.makedirs(self.tmp_path)
            self.save_trainval_split()
            self.DeepSDM_conf.experiment_id = self.logger.name
            self.DeepSDM_conf.run_id = self.logger.run_id
            
            with open(f'{self.tmp_path}/DeepSDM_conf.yaml', 'w') as f:
                yaml.dump(vars(self.DeepSDM_conf), f)
                
            self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = f'{self.tmp_path}/DeepSDM_conf.yaml', artifact_path = 'conf')
            self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = self.DeepSDM_conf.meta_json_files['env_inf'], artifact_path = 'conf')
            self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = self.DeepSDM_conf.meta_json_files['sp_inf'], artifact_path = 'conf')
            self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = self.DeepSDM_conf.meta_json_files['k_inf'], artifact_path = 'conf')
            self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = self.DeepSDM_conf.meta_json_files['co_vec'], artifact_path = 'conf')
            self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = self.DeepSDM_conf.geo_extent_file, artifact_path = 'conf')
            
    def on_fit_end(self):
        if self.trainer.global_rank == 0:
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    top_k_avg_state_dict = None
                    for model_path, monitered_value in cb.best_k_models.items():
                        
                        state_dict = torch.load(model_path)['state_dict']
                        if top_k_avg_state_dict is None:
                            top_k_avg_state_dict = state_dict
                        else:
                            for key in state_dict:
                                top_k_avg_state_dict[key] = top_k_avg_state_dict[key] + state_dict[key]
                    print(f'there are {len(cb.best_k_models.items())} models.')
                    for key in top_k_avg_state_dict:
                        top_k_avg_state_dict[key] = top_k_avg_state_dict[key] / len(cb.best_k_models.items())
        
                    torch.save(top_k_avg_state_dict, f'{self.tmp_path}/top_k_avg_state_dict.pt')
                    self.logger.experiment.log_artifact(run_id = self.logger.run_id, 
                                                        local_path = f'{self.tmp_path}/top_k_avg_state_dict.pt', 
                                                        artifact_path = 'top_k_avg_state_dict')
                
                    top_k_avg_model = LitUNetSDM(yaml_conf=self.yaml_conf, tmp_path=self.tmp_path)
                    top_k_avg_model.load_state_dict(top_k_avg_state_dict)

                    with mlflow.start_run(self.logger.run_id) as run:
                        mlflow.pytorch.log_model(top_k_avg_model, 'top_k_avg_model')
                    # print()
     
    
    
    def save_trainval_split(self):
        
        dataset = self.trainer.datamodule.dataset_train
        partition_extent = dataset.split_tif[:dataset.height_original, :dataset.width_original].numpy()
        
        fig_ = plt.figure()
        plt.imshow(self.trainer.datamodule.geo_extent.squeeze() * (1 + partition_extent.squeeze()) * .25)
        self.logger.experiment.log_figure(figure = fig_, artifact_file = 'train_val_geo_extent.png', run_id = self.logger.run_id)
        plt.close()
        
        tmp_output_file = f'{self.tmp_path}/partition_extent.tif'
        with rasterio.open(tmp_output_file,
                           'w', 
                           crs = self.trainer.datamodule.geo_crs,
                           transform = self.trainer.datamodule.geo_transform,
                           height = dataset.height_original, 
                           width = dataset.width_original, 
                           count = 1, 
                           nodata = 0, 
                           dtype = rasterio.int16) as img_towrite:
            
            img_towrite.write(partition_extent, 1)
        self.logger.experiment.log_artifact(run_id = self.logger.run_id, local_path = tmp_output_file, artifact_path = 'extent_binary')
