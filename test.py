import torch
import argparse
import time
from datetime import datetime
import importlib
from seren.utils.data import Interactions, Categories
from seren.utils.utils import get_logger, ACC_KPI
from seren.utils.model_selection import fold_out, train_test_split, handle_adj, build_graph
from seren.utils.dataset import NARMDataset, SRGNNDataset, GRU4RECDataset, ConventionDataset, GCEDataset
from seren.utils.metrics import accuracy_calculator, diversity_calculator, performance_calculator
from seren.model.narm import NARM
from seren.model.stamp import STAMP
from seren.model.mcprn_v4_block import MCPRN

from seren.model.gcegnn import CombineGraph
from seren.model.hide import HIDE
from seren.model.attenMixer import AreaAttnModel
from seren.model.conventions import Pop, SessionPop, ItemKNN, BPRMF, FPMC, PopNew
from seren.model.sknn import SessionKNN
from seren.utils.functions import reindex, get_dataloader
from seren.Best_setting import Best_setting
from config import Model_setting, HyperParameter_setting, Dataset_setting, Best_setting
import json
import os
import numpy as np
import optuna
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='games', help='amazon/games/ml1m')
parser.add_argument('--model', default='FPMC', help='MCPRN/STAMP/NARM/GCE-GNN/FPMC/HIDE')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--topK', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--trials', type=int, default=30)
parser.add_argument('--tune', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)


opt = parser.parse_args()


if __name__ == "__main__":
    if opt.dataset == 'amazon':
        train_data = np.load('seren/dataset/amz/train_amz_150.npy', allow_pickle=True).tolist()
        test_data = np.load('seren/dataset/amz/test_amz.npy', allow_pickle=True).tolist()
        candidate_data = np.load(f'seren/dataset/amz/candidate_test_{opt.seed}_amz.npy', allow_pickle=True).tolist()
    elif opt.dataset == 'games':
        train_data = np.load('seren/dataset/games/train_games_150.npy', allow_pickle=True).tolist()
        test_data = np.load('seren/dataset/games/test_games.npy', allow_pickle=True).tolist()
        candidate_data = np.load(f'seren/dataset/games/candidate_test_{opt.seed}_games.npy', allow_pickle=True).tolist()
    elif opt.dataset == 'cd':
        train_data = np.load('seren/dataset/cd/train_cd_150.npy', allow_pickle=True).tolist()
        test_data = np.load('seren/dataset/cd/valid_cd.npy', allow_pickle=True).tolist()
        candidate_data = np.load('seren/dataset/cd/candidate_vali.npy', allow_pickle=True).tolist()
    elif opt.dataset == 'ml1m':
        train_data = np.load('seren/dataset/ml1m/train_ml1m_150.npy', allow_pickle=True).tolist()
        test_data = np.load('seren/dataset/ml1m/test_ml1m.npy', allow_pickle=True).tolist()
        candidate_data = np.load(f'seren/dataset/ml1m/candidate_test_{opt.seed}_ml1m.npy', allow_pickle=True).tolist()

    model = SessionKNN(train_data, test_data, {'n': 50}, None)
    model.fit(train_data)

    for k in [1,5,10]:
        preds, last_items = model.predict(test_data, k=k, candidate=candidate_data)
        preds = torch.tensor(preds)
        last_items = torch.tensor(last_items)
        # print(preds.shape)
        # print(last_items.shape)
        metrics = accuracy_calculator(preds, last_items, ACC_KPI)
        print(metrics)  
    # model = PopNew(pop_n=100, logger=None)
    # model.fit(train_data)
    # for k in [1,5,10]:
    #     preds, last_items = model.predict(test_data, k=k)
    #     metrics = accuracy_calculator(preds, last_items, ACC_KPI)
    #     #['ndcg', 'mrr', 'hr']
    #     print(metrics)

    # preds, last_item = torch.LongTensor([]), torch.LongTensor([])
    # for idx, i in enumerate(test_data):
    #     pred = torch.tensor([candidate_data[idx]])
    #     target = [i[-1]]
    #     preds = torch.cat((preds, pred), 0)
    #     last_item = torch.cat((last_item, torch.tensor(target)), 0)
    # for k in [1,5,10]:
    #     # preds, last_items = model.predict(test_data, k=k)
    #     metrics = accuracy_calculator(preds, last_item, ACC_KPI)
    #     #['ndcg', 'mrr', 'hr']
    #     print(metrics)
