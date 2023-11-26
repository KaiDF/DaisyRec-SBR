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
from seren.model.conventions import Pop, SessionPop, ItemKNN, BPRMF, FPMC
from seren.utils.functions import reindex, get_dataloader
from seren.Best_setting import Best_setting
from config import Model_setting, HyperParameter_setting, Dataset_setting, Best_setting
import json
import os
import numpy as np
import optuna
import csv

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon', help='amazon/games/ml1m')
parser.add_argument('--model', default='FPMC', help='MCPRN/STAMP/NARM/GCE-GNN/FPMC/HIDE')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--topK', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--trials', type=int, default=30)
parser.add_argument('--tune', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)


opt = parser.parse_args()
init_seed(opt.seed)

def test():
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


    model_config = Model_setting[opt.model]
    data_config = Dataset_setting[opt.dataset]
    best_settings = Best_setting[opt.model][opt.dataset]
    model_config = {**model_config, **best_settings}
    model_config['gpu'] = opt.gpu
    logger = get_logger(f'tune_{model_config["description"]}_{opt.dataset}')
    
    # dataloader = importlib.import_module('seren.utils.dataset.{}'.format(model_config['dataloader']))
    dataloader = getattr(importlib.import_module('seren.utils.dataset'), model_config['dataloader'], None)
    train_dataset = dataloader(train_data, model_config)
    valid_dataset = dataloader(test_data, model_config, candidate_set=candidate_data, isTrain=False)
    

    if opt.model in ['NARM','FPMC','STAMP','MCPRN']:
        train_dataset = train_dataset.get_loader(model_config, shuffle=True)
        valid_dataset = valid_dataset.get_loader(model_config, shuffle=False)
        model = getattr(importlib.import_module('seren.model.{}'.format(model_config['model_dir'])), opt.model, None)
        model = model(data_config['num_node'], model_config, logger)

    elif opt.model in ['GCE-GNN']:
        adj, num = build_graph(train_data, data_config, model_config)
        num_node = data_config['num_node'] + 1
        adj, num = handle_adj(adj, num_node, model_config['n_sample_all'], num)
        model = CombineGraph(model_config, num_node, adj, num, logger)
    elif opt.model in ['HIDE']:
        num_node = data_config['num_node'] + 1
        model = HIDE(model_config, num_node, logger=logger)
    elif opt.model in ['AttenMixer']:
        train_dataset = train_dataset.get_loader(model_config, shuffle=True)
        valid_dataset = valid_dataset.get_loader(model_config, shuffle=False)
        num_node = data_config['num_node'] + 1
        model = AreaAttnModel(model_config, num_node, logger)

    
    # training process
    model.fit(train_dataset)#, valid_dataset)

    res_dir = f'res/sample150/{opt.dataset}/'
    f = open(res_dir + f'result_{opt.dataset}_{opt.model}_{opt.seed}.txt', 'a')
    for k in [1,5,10]:
        # print(k)
        line = f'HR@{k}\tNDCG@{k}\tMAP@{k}\n'
        f.write(line)
        preds, truth = model.predict(valid_dataset, k=k)
        # print(preds[:10])
        # print(truth[:10])
        # print(truth.shape)
        metrics = accuracy_calculator(preds, truth, ACC_KPI)
        res_line = f'{metrics[2]:.4f}\t{metrics[0]:.4f}\t{metrics[1]:.4f}\n'
        f.write(res_line)
        f.flush()
    f.close()
        # ['ndcg', 'mrr', 'hr']
        # print(metrics)

TRIAL_CNT = 0
def tune():
    # global TRIAL_CNT
    global train_dataset
    global valid_dataset
    global candidate_data

    if opt.dataset == 'amazon':
        train_data = np.load('seren/dataset/amz/train_amz_150.npy', allow_pickle=True).tolist()
        vali_data = np.load('seren/dataset/amz/valid_amz.npy', allow_pickle=True).tolist()
        candidate_data = np.load('seren/dataset/amz/candidate_vali.npy', allow_pickle=True).tolist()
    elif opt.dataset == 'games':
        train_data = np.load('seren/dataset/games/train_games_150.npy', allow_pickle=True).tolist()
        vali_data = np.load('seren/dataset/games/valid_games.npy', allow_pickle=True).tolist()
        candidate_data = np.load('seren/dataset/games/candidate_vali.npy', allow_pickle=True).tolist()
    elif opt.dataset == 'cd':
        train_data = np.load('seren/dataset/cd/train_cd_150.npy', allow_pickle=True).tolist()
        vali_data = np.load('seren/dataset/cd/valid_cd.npy', allow_pickle=True).tolist()
        candidate_data = np.load('seren/dataset/cd/candidate_vali.npy', allow_pickle=True).tolist()
    elif opt.dataset == 'ml1m':
        train_data = np.load('seren/dataset/ml1m/train_ml1m_150.npy', allow_pickle=True).tolist()
        vali_data = np.load('seren/dataset/ml1m/valid_ml1m.npy', allow_pickle=True).tolist()
        candidate_data = np.load('seren/dataset/ml1m/candidate_vali.npy', allow_pickle=True).tolist()


    model_config = Model_setting[opt.model]
    model_config['gpu'] = opt.gpu
    data_config = Dataset_setting[opt.dataset]
    logger = get_logger(f'tune_{model_config["description"]}_{opt.dataset}')
    
    # dataloader = importlib.import_module('seren.utils.dataset.{}'.format(model_config['dataloader']))
    dataloader = getattr(importlib.import_module('seren.utils.dataset'), model_config['dataloader'], None)
    train_dataset = dataloader(train_data, model_config)
    valid_dataset = dataloader(vali_data, model_config, candidate_set=candidate_data, isTrain=False)
    if opt.model in ['NARM','FPMC','STAMP','MCPRN', 'AttenMixer']:
        train_dataset = train_dataset.get_loader(model_config, shuffle=True)
        valid_dataset = valid_dataset.get_loader(model_config, shuffle=False)

    tune_params = []
    def objective(trial):
        global TRIAL_CNT
        for key, value in HyperParameter_setting[opt.model].items():
            if key == 'int':
                for para_name, scales in value.items():
                    model_config[para_name] = trial.suggest_int(para_name, scales['min'], scales['max'], step=scales['step'])
                    tune_params.append(para_name)
            elif key == 'categorical':
                for para_name, scales in value.items():
                    model_config[para_name] = trial.suggest_categorical(para_name, scales)
                    tune_params.append(para_name)
    

        if opt.model in ['NARM','FPMC','STAMP','MCPRN']:
            model = getattr(importlib.import_module('seren.model.{}'.format(model_config['model_dir'])), opt.model, None)
            model = model(data_config['num_node'], model_config, logger)

        elif opt.model in ['GCE-GNN']:
            adj, num = build_graph(train_data, data_config, model_config)
            num_node = data_config['num_node'] + 1
            adj, num = handle_adj(adj, num_node, model_config['n_sample_all'], num)
            model = CombineGraph(model_config, num_node, adj, num, logger)
        elif opt.model in ['HIDE']:
            num_node = data_config['num_node'] + 1
            model = HIDE(model_config, num_node, logger=logger)
        elif opt.model in ['AttenMixer']:
            # train_dataset = train_dataset.get_loader(model_config, shuffle=True)
            # valid_dataset = valid_dataset.get_loader(model_config, shuffle=False)
            num_node = data_config['num_node'] + 1
            model = AreaAttnModel(model_config, num_node, logger)

    
        # training process
        model.fit(train_dataset)#, valid_dataset)
        preds, truth = model.predict(valid_dataset, k=opt.topK)
        metrics = accuracy_calculator(preds, truth, ACC_KPI)
        # ['ndcg', 'mrr', 'hr']
        kpi = metrics[0]
        logger.info(f"Finish {TRIAL_CNT+1} trial for {opt.model}...")
        TRIAL_CNT += 1

        return kpi

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=opt.seed))
    study.optimize(objective, n_trials=opt.trials)

    tune_params = list(set(tune_params))
    tune_log_path = f'./tune_log/sample_150/{opt.dataset}/'
    res_csv = tune_log_path + f'result_{opt.dataset}_{opt.model}.csv'
    with open(res_csv, 'w', newline='') as f:
        fieldnames = ['Trial ID'] + tune_params + ['NDCG@5']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for estudio in study.trials:
            w_dict = {}
            w_dict['Trial ID'] = estudio.number+1
            for paras in tune_params:
                w_dict[paras] = estudio.params[paras]
            w_dict['NDCG@5'] = estudio.value
            writer.writerow(w_dict)

        best_dict = {}
        best_dict['Trial ID'] = study.best_trial.number+1
        best_dict['NDCG@5'] = study.best_value
        for paras in tune_params:
            best_dict[paras] = study.best_trial.params[paras]
        writer.writerow(best_dict)
        f.flush()
        f.close()

    logger.info(f"Best trial for {opt.model}: {study.best_trial.number+1}")



if __name__ == '__main__':
    if opt.tune:
        tune()
    elif opt.test:
        test()




