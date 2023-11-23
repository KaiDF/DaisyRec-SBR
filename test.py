import torch
from seren.utils.dataset import AttMixerDataset
import numpy as np
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler, SequentialSampler
from seren.utils.functions import reindex, get_dataloader

if __name__ == "__main__":
    def train_dataloader(data):
        sampler = BatchSampler(SequentialSampler(data), batch_size=32, drop_last=False)
        
        return DataLoader(data, sampler=sampler, num_workers=4, pin_memory=True)
    def get_dataloader1(data):
        sampler = BatchSampler(SequentialSampler(data), batch_size=32, drop_last=False)
        return DataLoader(data, batch_sampler=sampler, num_workers=4, pin_memory=True)
    conf = {}
    conf['session_len'] = 50
    conf['batch_size'] = 32
    # data = SessionData()
    # data.setup()
    train_data = np.load('seren/dataset/amz/train_amz.npy', allow_pickle=True).tolist()
    # train_data = pickle.load(open('../data/amz/train.txt', 'rb'))
    # train_data = Data(train_data, shuffle=True)
    # val_loader = data.val_dataloader()
    train_data = AttMixerDataset(train_data, conf)
    train_loader = get_dataloader1(train_data)
    train_loader1 = train_dataloader(train_data)
    
    for i in train_loader:
        # a = i[-2].shape
        print(i)
        break
    for i in train_loader1:
        # a = i[-2].shape
        print(i)
        break