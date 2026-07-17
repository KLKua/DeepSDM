import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from types import SimpleNamespace
import os

class TaxaDataset(Dataset):
    def __init__(self, env_stack, embedding, label_stack, k2_stack, trainorval, DeepSDM_conf, cuda_id=0):

        self.cuda_id = cuda_id
        
        self.species_date_list = label_stack['species_date']
        self.species_list = label_stack['species']
        self.date_list = label_stack['date']
        self.embedding = embedding
        self.split = torch.tensor(np.loadtxt(os.path.join('./workspace', 'partition.txt'), delimiter = ',')).to(torch.int)
        self.extent_split = torch.tensor(np.loadtxt(os.path.join('./workspace', 'extent_partition.txt'), delimiter = ',')).to(torch.int)
        self.training_conf = SimpleNamespace(**DeepSDM_conf.training_conf)
        reproducibility_conf = getattr(DeepSDM_conf, 'reproducibility_conf', {})
        self.seed = int(reproducibility_conf.get('seed', 42))
        self.current_epoch = 0

        if trainorval == 'train':
            self.trainorval = 1
            self.random_stack_num = self.training_conf.num_train_subsample_stacks
        else:
            self.trainorval = 0
            self.random_stack_num = self.training_conf.num_val_subsample_stacks
        self.stage_seed_offset = 0 if trainorval == 'train' else 1_000_000
        self.crop_epoch_multiplier = 1_000_003 if trainorval == 'train' else 0
        
        self.height_original = label_stack['tensor'].shape[1]
        self.width_original = label_stack['tensor'].shape[2]
        
        # if 'height_original' is divisible by the height number of split(aka split.shape[0])
        # 'height_new' = 'height_original'
        # otherwise 'height_new' will be 'height_original' adding a smallest number to be the closest number that divisible by height number of split
        if self.height_original % self.split.shape[0] == 0:
            self.height_new = self.height_original
        else:
            self.height_new = self.height_original + (self.split.shape[0] - self.height_original % self.split.shape[0])
        # same situation of 'width_new' and 'width_original'
        if self.width_original % self.split.shape[1] == 0:
            self.width_new = self.width_original
        else:
            self.width_new = self.width_original + (self.split.shape[1] - self.width_original % self.split.shape[1])
        
        # train, val size based on split 
        self.split_height = self.height_new // self.split.shape[0]
        self.split_width = self.width_new // self.split.shape[1]

#         torch.cuda.synchronize()
#         starttime = time.time()
#         print('########## STACKS ##########')

        # subsample size based on split
        split_tif = torch.zeros(self.height_new, self.width_new)
        split_element = []
        h_, w_ = torch.where((self.split == self.trainorval) & (self.extent_split == 1))
        for i in range(sum((self.split.view(-1) == self.trainorval) & (self.extent_split.view(-1) == 1))):
            height_start = h_[i] * self.split_height
            height_end = (h_[i] + 1) * self.split_height
            width_start = w_[i] * self.split_width
            width_end = (w_[i] + 1) * self.split_width
            split_tif[height_start : height_end, width_start : width_end] = 1
            split_element.append([height_start, height_end, width_start, width_end])
        self.split_tif = split_tif
        self.split_element = split_element
        
        # adjust shape of env_stack, label_stack and k2 to (height_new, width_new)
        # env_stack
        env_stack_new = env_stack['tensor']
        torch.cuda.synchronize()
        self.env_stack = F.pad(env_stack_new,
                              (0, (self.width_new - self.width_original), 0, (self.height_new - self.height_original)),
                              mode = 'replicate')
        self.env_stack = self.env_stack.cpu()
        # torch.cuda.empty_cache()
        
        #label_stack
        label_stack_new = label_stack['tensor']
        self.label_stack = F.pad(label_stack_new,
                                 (0, (self.width_new - self.width_original), 0, (self.height_new - self.height_original)), 
                                 mode = 'constant', 
                                 value = 0)
        self.label_stack = self.label_stack.cpu()
        # torch.cuda.empty_cache()
        
        #k2
        k2_stack_new = k2_stack['tensor']
        self.k2_stack = F.pad(k2_stack_new,
                              (0, (self.width_new - self.width_original), 0, (self.height_new - self.height_original)), 
                              mode = 'constant',
                              value = -9999)
        self.k2_stack = self.k2_stack.cpu()
        # torch.cuda.empty_cache()

        self.k2_stack_date = k2_stack['date']

#         torch.cuda.synchronize()
#         print(time.time() - starttime)
#         print('########## STACKS ##########')
        self.filter_split_element()

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _deterministic_crop(self, tensor, index):
        th = self.training_conf.subsample_height
        tw = self.training_conf.subsample_width
        h, w = tensor.shape[-2:]

        if h < th or w < tw:
            raise ValueError(f'Crop size ({th}, {tw}) is larger than tensor size ({h}, {w}).')
        if h == th and w == tw:
            return tensor

        generator = torch.Generator()
        generator.manual_seed(
            self.seed
            + self.stage_seed_offset
            + self.crop_epoch_multiplier * self.current_epoch
            + int(index)
        )
        top = torch.randint(0, h - th + 1, (1,), generator=generator).item()
        left = torch.randint(0, w - tw + 1, (1,), generator=generator).item()

        return tensor[..., top:top + th, left:left + tw]
        
    def __getitem__(self, index):
        
        # idx_species_date, idx_split = self._getidx(index)
        pair_idx = index // self.random_stack_num
        idx_species_date, idx_split = self.valid_pairs[pair_idx]
        height_start, height_end, width_start, width_end = self._getextent(idx_split)
        
        # embeddings
        species = self.species_list[idx_species_date]
        date = self.date_list[idx_species_date]
        embeddings = torch.tensor(self.embedding[species]).reshape(-1, 1, 1)
        
        # k
        idx_date = self.k2_stack_date.index(date)
        k2 = self.k2_stack[idx_date:(idx_date+1), height_start:height_end, width_start:width_end]#.cuda(self.cuda_id)
        
        # inputs
        inputs = self.env_stack[idx_date, :, height_start:height_end, width_start:width_end]#.cuda(self.cuda_id)
        
        # labels
        labels = self.label_stack[idx_species_date:(idx_species_date+1), height_start:height_end, width_start:width_end]#.cuda(self.cuda_id)
            
        embeddings = torch.unsqueeze(embeddings, axis=0)
        k2 = torch.unsqueeze(k2, axis=0)
        inputs = torch.unsqueeze(inputs, axis=0)
        labels = torch.unsqueeze(labels, axis=0)

        stacked_all = torch.cat([inputs, labels, k2], axis = 1)
        stacked_all = self._deterministic_crop(stacked_all, index)
        inputs_transform = stacked_all[:, 0:inputs.shape[1]]
        labels_transform = stacked_all[:, inputs.shape[1]:(inputs.shape[1] + labels.shape[1])]
        k2_transform = stacked_all[:, (inputs.shape[1] + labels.shape[1]):(inputs.shape[1] + labels.shape[1] + k2.shape[1])]

        return [inputs_transform, embeddings], labels_transform, k2_transform, species, date

    def __len__(self):
        return len(self.valid_pairs) * self.random_stack_num
    
    
    def _getidx(self, index):
        idx_species_date = index // self.random_stack_num % len(self.species_date_list)
        idx_split = index // self.random_stack_num // len(self.species_date_list)

        return idx_species_date, idx_split
    
    def _getextent(self, idx_split):
        height_start = self.split_element[idx_split][0]
        height_end = self.split_element[idx_split][1]
        width_start = self.split_element[idx_split][2]
        width_end = self.split_element[idx_split][3]
        return height_start, height_end, width_start, width_end

    # def filter_split_element(self):
    #     self.valid_pairs = []          # [(idx_species_date, idx_split), ...]
    #     for split_idx, (h0, h1, w0, w1) in enumerate(self.split_element):
    #         # 取出這個 split 區塊的所有物種日期 label
    #         patch = self.label_stack[:, h0:h1, w0:w1]
    #         # 查看每個 species_date 是否至少有一個非零像素
    #         non_zero = patch.flatten(1).any(dim=1)    # shape = (num_species_date,)
    #         for sp_idx, has_label in enumerate(non_zero.tolist()):
    #             if has_label:                         # 至少一個標籤≠0
    #                 self.valid_pairs.append((sp_idx, split_idx))

                    
    def filter_split_element(self):
        self.valid_pairs = []          # [(idx_species_date, idx_split), ...]
        # 建立 species_date 對應的日期索引對照表，方便查詢
        date_to_idx = {d: i for i, d in enumerate(self.k2_stack_date)}
        
        for split_idx, (h0, h1, w0, w1) in enumerate(self.split_element):
            # 先取出這個 split 範圍的 k2 所有日期資料 (num_date, H, W)
            k2_patch = self.k2_stack[:, h0:h1, w0:w1]
            # 針對每個 species_date
            for sp_idx, date in enumerate(self.date_list):
                if date not in date_to_idx:
                    continue  # 防止找不到日期索引
                
                idx_date = date_to_idx[date]
                k2_subpatch = k2_patch[idx_date]   # (H, W)
                # 判斷 k2_subpatch 是否有 > 0
                if (k2_subpatch > 0).any():
                    self.valid_pairs.append((sp_idx, split_idx))
