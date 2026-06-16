import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import umap

class CreateDataset(Dataset):
    def __init__(self, train, label):
        self.feature_ = train
        self.label_ = label

    def __len__(self):
        # Return size of dataset
        return len(self.feature_)

    def __getitem__(self, idx):
        return torch.tensor(self.feature_[idx], dtype = torch.long), torch.tensor(self.label_[idx])
    

class EmbeddingModel(nn.Module):

    def __init__(self, num_species, num_vector):
        super().__init__()
        self.species_vectors = nn.Embedding(num_species, num_vector)
        nn.init.xavier_uniform_(self.species_vectors.weight)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, pos_idxs, ys, neg_idxs, num_neg=10):
        #
        # Compute positive samples
        # ----------------------------------------------------------------
        # u,v: [batch_size, emb_dim]
        u = self.species_vectors(pos_idxs[0]) # idx of sp1
        v = self.species_vectors(pos_idxs[1]) # idx of sp2
        alpha = torch.log(torch.pow(ys, 2) + 1.) + 1.
        positive_loss = -alpha * self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()

        #
        # Compute negative samples
        # ----------------------------------------------------------------
        nu = self.species_vectors(neg_idxs[0])
        nv = self.species_vectors(neg_idxs[1])
        negative_loss = -self.log_sigmoid(-torch.sum(nu * nv, dim=1)).squeeze()

        return (torch.sum(positive_loss) + torch.sum(negative_loss)) / (pos_idxs.shape[1] * (1 + num_neg))

####################################################################################
# TODO: NEED VERY HEAVY REFACTORING
# ###################################################################################
class TrainEmbedding:
    def __init__(self, embedding_conf, CreateDataset, EmbeddingModel, 
                                    output_dir='./workspace/species_data',
                                    idx2species_file='idx2species.csv',
                                    cooccurrence_counts_file='cooccurrence.csv'):
        super().__init__()
        
        self.batch_size = embedding_conf.batch_size
        self.num_vector = embedding_conf.num_vector
        self.num_neg = embedding_conf.num_neg
        self.epochs = embedding_conf.epochs
        self.CreateDataset = CreateDataset
        self.EmbeddingModel = EmbeddingModel

        self.output_dir = output_dir
        self.embedding_dir = os.path.join(self.output_dir, 'embedding')
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        
        self.idx2species_file = idx2species_file
        self.cooccurrence_counts_file = cooccurrence_counts_file

    def _get_idx_species(self):
        
        idx2species = dict()
        species2idx = dict()
        Xs = []
        ys = []
        nXs = []

        # trashmai changed the cooccurrence output to sp combos with 0 count, for a full species list
        # othewise, the idx2sp will only have species name with > 0 counts
        matrix = pd.read_csv(os.path.join(self.output_dir, f'cooccurrence_data/{self.cooccurrence_counts_file}'), sep = '\t')
        for i, lines in matrix.iterrows() :
            sp1 = lines['sp1']
            sp2 = lines['sp2']
            cooccur_counts = lines['counts']
            if sp1 == sp2:
                continue
                
            # ~!@#$%^&*()_+
            if sp1 not in species2idx:
                species2idx[sp1] = len(idx2species)
                idx2species[len(idx2species)] = sp1
            if sp2 not in species2idx:
                species2idx[sp2] = len(idx2species)
                idx2species[len(idx2species)] = sp2

            # positive samples
            if cooccur_counts > 0:
                Xs.append((species2idx[sp1], species2idx[sp2]))
                ys.append(float(cooccur_counts))
            else:
                nXs.append((species2idx[sp1], species2idx[sp2]))
            
        with open(os.path.join(self.embedding_dir, self.idx2species_file), 'w') as out_file:
            for key in sorted(idx2species.keys()):
                try:
                    out_file.write(f'{key}\t{idx2species[key]}\n')
                except:
                    print(key, idx2species[key])

        print(f'embedding size = ({len(idx2species)}, {self.num_vector})')
        print(f'number of data = positive: {len(Xs)} + negative: {len(nXs)}')
        print(f'batch_size = {self.batch_size}; num_neg = {self.num_neg}')

        self.idx2species = idx2species
        self.Xs = Xs
        self.ys = ys
        self.nXs = np.array(nXs)


    def setup(self):

        self._get_idx_species()
        self.dataset = self.CreateDataset(self.Xs, self.ys)
                  
        # Model training
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.EM = self.EmbeddingModel(len(self.idx2species), self.num_vector)
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.optimizer = torch.optim.AdamW(self.EM.parameters())  # learning rate
                  
    def train(self):
        
        min_loss = np.inf
        patience_count = 0
        patience = 500

        self.EM.to(self.dev)
        self.loss_to_log = []
        for epoch in range(1, self.epochs + 1):
            cum_loss = 0.
            for (train_batch, label_batch) in self.train_dataloader:

                # Negative sampling for column
#                 neg_smpls = np.zeros(self.num_neg)
#                 for i in range(train_batch.shape[0]):
#                     delta = random.sample(list(range(len(self.idx2species))), self.num_neg)
#                     neg_smpls = np.vstack([neg_smpls, delta])
#                 neg_cols = torch.tensor(neg_smpls[1:].reshape(-1), dtype=torch.long)
#                 neg_rows = train_batch[:, 0].repeat(self.num_neg)
#                 neg_idxs = torch.vstack([neg_rows, neg_cols])
                
                # minor expelling each other in the cooccurrence groups
                pseudo_neg1_smpls = []
                for i in range(train_batch.shape[0]):
                    delta = random.sample(range(len(self.idx2species)), self.num_neg)
                    pseudo_neg1_smpls.extend(delta)
                pseudo_neg1_idxs = torch.vstack([train_batch[:, 0].repeat(self.num_neg), torch.tensor(pseudo_neg1_smpls)])
                
                # minor expelling each other in the non-cooccurrence groups
                pseudo_neg2_idxs = torch.from_numpy(self.nXs[random.sample(range(self.nXs.shape[0]), min(self.nXs.shape[0], self.num_neg * train_batch.shape[0]))].reshape(2, -1))

                neg_idxs = torch.hstack([pseudo_neg1_idxs, pseudo_neg2_idxs])

                # Model Prediction and loss calculation
                loss = self.EM(train_batch.T.to(self.dev), label_batch.to(self.dev), neg_idxs.to(self.dev), self.num_neg)
                
                # Model updating
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.EM.parameters(), 5)    # gradient clipping
                self.optimizer.step()
                cum_loss += loss.item()
            
            loss_avg = cum_loss / len(self.train_dataloader)
            print(f'Epoch: {epoch}, loss: {loss_avg:.04f}', end='\r')
            self.loss_to_log.append(loss_avg)
            if loss_avg < min_loss:
                min_loss = loss_avg
                patience_count = 0
                torch.save(self.EM.state_dict(), os.path.join(self.embedding_dir, f'embedding_best.pt'))
            else:
                patience_count += 1

            if patience_count >= patience:
                print('Early stopped!')
                break

            if epoch % 500 == 0:
                pd.DataFrame(
                    {'epoch': list(range(1, epoch + 1)), 
                     'loss': self.loss_to_log}).to_csv(os.path.join(self.embedding_dir, 'embedding_loss.csv'), index = None)
                # torch.save(self.EM.state_dict(), os.path.join(self.embedding_dir, f'embedding_all_{epoch}.pt'))

        pd.DataFrame(
            {'epoch': list(range(1, epoch + 1)), 
                'loss': self.loss_to_log}).to_csv(os.path.join(self.embedding_dir, 'embedding_loss.csv'), index = None)


    def log_embedding_result(self, model_pth='embedding_best.pt', cooccurrence_json_file_out='./workspace/cooccurrence_vector.json'):
        
        model = self.EM
        model.load_state_dict(torch.load(os.path.join(self.embedding_dir, model_pth), map_location = self.dev))

        # result
        embedding_result = dict()
        id2sp = pd.read_csv(os.path.join(self.embedding_dir, self.idx2species_file), header = None, sep = '\t')
        for i, s in id2sp.iterrows():
            embedding_result[s[1]] = model.to(self.dev).species_vectors(torch.tensor(i).to(self.dev)).tolist()
        
        with open(cooccurrence_json_file_out, 'w') as f:
            json.dump(embedding_result, f)
        
        self.embedding = embedding_result
    
    def visualize_embedding(self, embedding=None, show_txt=False, n_neighbors=15):
        list_all = []
        sp_all = []
        if embedding is None:
            embedding = self.embedding
        for sp in sorted(embedding.keys()):
            list_all.append(embedding[sp])
            sp_all.append(sp)
        df = pd.DataFrame(list_all)

        reducer = umap.UMAP(n_neighbors=15)
        embedding_umap = reducer.fit_transform(df.values)

        fig, ax = plt.subplots()
        ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1])
        if show_txt:
            for i, txt in enumerate(sp_all):
                ax.annotate(txt, (embedding_umap[i, 0], embedding_umap[i, 1]))
