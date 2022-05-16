import os
import json
import time
import argparse

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from node2vec_impl_dp import Node2Vec

# -- Global Variables --
NUM_EPOCHS = 30
BATCH_SIZE = 32
P=1
Q=1
WALK=50
LR=0.0025
SAVE_PATH = 'results/'
DATA_PATH = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Geometric Node2Vec Training on Reddit.')
    parser.add_argument('--gpu', default=1, help='number of gpus')
    parser.add_argument('--type', default='v100', help='type of gpu')
    args = parser.parse_args()
    NUM_GPUS = int(args.gpu)
    NUM_GPUS_STR = str(NUM_GPUS)
    TYPE = args.type

    
    # load index map
    with open(f'{DATA_PATH}reddit_index.json') as f:
        reddit_dict = json.load(f)

    # read reddit s2d as a Data object in pytorch geometric
    ## read the entire dataset in memory as a pandas dataframe
    df = pd.read_csv(f'{DATA_PATH}reddit_subreddit_to_domain__gt-01-urls.csv', header=None)
    
    ## extract source and target nodes and map to corresponding integer indices
    source_nodes = df.iloc[:,0].apply(lambda x: reddit_dict[x]).values.tolist()
    target_nodes = df.iloc[:,1].apply(lambda x: reddit_dict[x]).values.tolist()
    num_nodes = len(set(source_nodes).union(set(target_nodes)))
    weight = df.iloc[:,2].values.tolist()

    ## convert to pytorch geometric Data object
    edge_index = torch.tensor([source_nodes, target_nodes])
    edge_attr = torch.tensor(weight)[:,None]
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes
    transform = T.ToUndirected()
    data = transform(data)
    
    # read domain ideology for evaluation
    domain_ideology = pd.read_csv(f'{DATA_PATH}robertson_et_al.csv')
    domain_ideology = domain_ideology[['domain', 'score']].copy()
    domain_ideology['id'] = domain_ideology['domain'].apply(lambda x: reddit_dict[x] if x in reddit_dict else None)
    domain_ideology = domain_ideology[domain_ideology['id'].notna()].reset_index(drop=True)
    domain_ideology['id'] = domain_ideology['id'].astype('int64')
    
    # train, test, val split
    train = domain_ideology.sample(frac=0.8,random_state=42)
    test = domain_ideology[~domain_ideology.index.isin(train.index)]
    train_sub = train.sample(frac=0.8, random_state=24)
    val = train[~train.index.isin(train_sub.index)]

    train_x, train_y = train_sub['id'].tolist(), train_sub['score'].tolist()
    val_x, val_y = val['id'].tolist(), val['score'].tolist()
    
    # model specification
    model = Node2Vec(data.edge_index, embedding_dim=128, 
              walk_length=WALK, context_size=10, walks_per_node=10, 
              num_negative_samples=1, p=P, q=Q, sparse=True)
    model = torch.nn.DataParallel(model) # wrap it as a data parallel object
    model.to(device)
    loader = model.module.loader(batch_size=BATCH_SIZE * NUM_GPUS, shuffle=True, num_workers=4)

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=LR)
    
    def train():
        """
        Train node2vec batch by batch using postive and negative samples from loader.
        Returns training loss (log-likelihood).
        """
        model.train()
        total_loss = 0

        fw_time = 0.0
        st = time.time()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            # concatenate along last dimension and transfer to GPU
            batch = torch.cat((pos_rw, neg_rw), -1).to(device)

            # for calling data parallel, call model.forward
            # for calling forward without dataparallel, call model.module
            train_st = time.time()
            loss = model(batch)
            loss = loss.sum()/NUM_GPUS

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            fw_time += time.time() - train_st

        return total_loss / len(loader), fw_time, time.time() - st

    @torch.no_grad()
    def test():
        """
        Evaluate embedding on downstream ideology scoring task using default predictor (Ridge).
        Returns train and validation MSE.
        """
        model.eval()
        z = model.module()

        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        clf = Ridge(alpha=0.01).fit(z[train_x].detach().cpu().numpy(), train_y)
        train_mse = mean_squared_error(train_y, clf.predict(z[train_x].detach().cpu().numpy()))

        preds = clf.predict(z[val_x].detach().cpu().numpy())
        val_mse = mean_squared_error(val_y, preds)

        return train_mse, val_mse
    
    # -- Main -- #
    # logging
    if os.path.exists(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_log.txt'):
        os.remove(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_log.txt')
    
    with open(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_log.txt', 'a') as f:
        f.write(f'Loss,Train MSE,Val MSE,Total Time,Train Time\n')

    loss_hist = []
    train_mse_hist = []
    val_mse_hist = []
    time_hist = []
    train_time_hist = []
    val_time_hist = []
    for epoch in range(1, NUM_EPOCHS):
        
        # -- Train -- #
        loss, train_time, total_time = train()

        # -- Validation -- #
        train_mse, val_mse = test()

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Total Time: {total_time/60:.2f} mins, Train time: {train_time/60:.2f} mins')
        
        # logging
        with open(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_log.txt', 'a') as f:
            f.write(f'{str(loss)},{str(train_mse)},{str(val_mse)},{str(end_time-start_time)},{str(train_time)}\n')

        # add to history
        loss_hist.append(loss)
        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        time_hist.append(end_time-start_time)
        train_time_hist.append(train_time)
        val_time_hist.append(val_time)

    # save
    loss_hist = np.array(loss_hist)
    train_mse_hist = np.array(train_mse_hist)
    val_mse_hist = np.array(val_mse_hist)
    time_hist = np.array(time_hist)
    train_time_hist = np.array(train_time_hist)
    val_time_hist = np.array(val_time_hist)

    np.save(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_loss.npy',loss_hist)
    np.save(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_train.npy',train_mse_hist)
    np.save(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_val.npy',val_mse_hist)
    np.save(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_total_time.npy',time_hist)
    np.save(f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}_train_time.npy',train_time_hist)

    torch.save(model.state_dict(), f'{SAVE_PATH}{TYPE}_{NUM_GPUS_STR}.pth')