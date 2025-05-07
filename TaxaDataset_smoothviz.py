import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TaxaDataset_smoothviz(Dataset):
    def __init__(self, idx_species_date, env_stack, embedding, label_stack, subsample_height, subsample_width, num_smoothviz_steps):
            
        self.species_date = label_stack['species_date'][idx_species_date] #e.g.'Acridotheres_cristatellus_2000-01-01'
        self.species = label_stack['species'][idx_species_date]
        self.date = label_stack['date'][idx_species_date]
        self.label = label_stack['tensor'][idx_species_date, ]#.cuda()

        self.embedding = torch.tensor(embedding[self.species]).reshape(-1, 1, 1)#.cuda()
        
        idx_date = env_stack['date'].index(self.date)
        env = env_stack['tensor'][idx_date, ]#.cuda()

        
        self.height_original = label_stack['tensor'].shape[1]
        self.width_original = label_stack['tensor'].shape[2]
        self.subsample_height = subsample_height #size of each subsample
        self.subsample_width = subsample_width
        
        self.height_elements = self.height_original // self.subsample_height # Number of subsample of original height (int)
        self.width_elements = self.width_original // self.subsample_width # Number of subsample of original width (int)
        
        self.height_new = self.height_original + 3 * self.subsample_height
        self.width_new = self.width_original + 3 * self.subsample_width
        self.num_smoothviz_steps = num_smoothviz_steps # Number of sets of original height adn width when smoothviz the img. Make sure this should be divided by subsample_size. e.g. if this number is 3, it means that the result of smotthvize is made by 3*3 imgs
        
        self.step_height = self.subsample_height // self.num_smoothviz_steps # length of every time smoothviz the img
        self.step_width = self.subsample_width // self.num_smoothviz_steps

        assert(self.step_height == self.subsample_height / self.num_smoothviz_steps)
        assert(self.step_width == self.subsample_width / self.num_smoothviz_steps)
        
        env_new = []
        for i in range(len(env)):
            env_new.append(env[i:(i+1), :, :].cuda())

#             env_new.append(F.pad(env[i:(i+1), :, :], 
#                                  (self.subsample_width, 2 * self.subsample_width, self.subsample_height, 2 * self.subsample_height), 
#                                  mode = 'replicate').to(torch.float))
        self.env = F.pad(torch.cat(env_new),
                         (self.subsample_width, 2 * self.subsample_width, self.subsample_height, 2 * self.subsample_height), 
                         mode = 'replicate') #.to(torch.float)
        self.env = self.env.cpu()
        torch.cuda.empty_cache()
        
    def __getitem__(self, index):
        idx_height_elements = index % (self.height_elements + 2)
        other_idx = index // (self.height_elements + 2)
        
        idx_width_elements = other_idx % (self.width_elements + 2)
        other_idx = other_idx // (self.width_elements + 2)
        
        idx_height_step = other_idx % self.num_smoothviz_steps
        idx_width_step = other_idx // self.num_smoothviz_steps
        
        step_height = self.step_height * idx_height_step
        step_width = self.step_width * idx_width_step
        
        height_start, height_end = idx_height_elements * self.subsample_height + step_height, (idx_height_elements + 1) * self.subsample_height + step_height
        width_start, width_end = idx_width_elements * self.subsample_width + step_width, (idx_width_elements + 1) * self.subsample_width + step_width
        
        inputs = self.env[:, height_start:height_end, width_start:width_end]#.cuda()

        return inputs, self.embedding, (height_start, height_end, width_start, width_end), self.species_date
    
    def __len__(self):
        return (self.height_elements + 2) * (self.width_elements + 2) * self.num_smoothviz_steps * self.num_smoothviz_steps

    def async_cuda(self):
        self.env = self.env.cuda(non_blocking=True)
        self.embedding = self.embedding.cuda(non_blocking=True)
#         return self
        
    def cpu(self):
        self.env = self.env.cpu()
        self.embedding = self.embedding.cpu()
#         return self