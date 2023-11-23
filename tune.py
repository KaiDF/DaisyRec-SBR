import torch
import argparse
import time
from datetime import datetime
import importlib
from seren.utils.data import Interactions, Categories
from seren.config import get_parameters, get_logger, ACC_KPI, DIV_KPI
from seren.utils.model_selection import fold_out, train_test_split, handle_adj, build_graph
from seren.utils.dataset import NARMDataset, SRGNNDataset, GRU4RECDataset, ConventionDataset, GCEDataset
from seren.utils.metrics import accuracy_calculator, diversity_calculator, performance_calculator
from seren.model.narm import NARM
from seren.model.stamp import STAMP
from seren.model.mcprn_v4_block import MCPRN
from seren.model.srgnn import SessionGraph
from seren.model.gcsan_v2 import GCSAN
from seren.model.gcegnn import CombineGraph
from seren.model.hide import HIDE
from seren.model.attenMixer import AreaAttnModel
from seren.model.gru4rec import GRU4REC
from seren.model.conventions import Pop, SessionPop, ItemKNN, BPRMF, FPMC
from seren.utils.functions import reindex, get_dataloader
from seren.config import TUNE_PATH
from config import Model_setting, HyperParameter_setting, Dataset_setting
import json
import os
import numpy as np
import optuna

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--model', default='MCPRN', help='MCPRN/STAMP/NARM/GCE-GNN/FPMC/HIDE')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--topK', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')


opt = parser.parse_args()

def test():
    init_seed(opt.seed)
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
    logger = get_logger(__file__.split('.')[0] + f'_{model_config["description"]}')
    
    # dataloader = importlib.import_module('seren.utils.dataset.{}'.format(model_config['dataloader']))
    dataloader = getattr(importlib.import_module('seren.utils.dataset'), model_config['dataloader'], None)
    train_dataset = dataloader(train_data, model_config)
    valid_dataset = dataloader(vali_data, model_config, candidate_set=candidate_data, isTrain=False)
    

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
    preds, truth = model.predict(valid_dataset, k=5)
    metrics = accuracy_calculator(preds, truth, ACC_KPI)
    print(metrics)

TRIAL_CNT = 0
if __name__ == '__main__':
    # main()
    init_seed(opt.seed)
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
    logger = get_logger(__file__.split('.')[0] + f'_{model_config["description"]}')
    
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
        global train_dataset
        global valid_dataset
        global candidate_data

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
        # hr, mrr, ndcg
        kpi = metrics[-1]
        logger.info(f"Finish {TRIAL_CNT+1} trial for {opt.model}...")
        TRIAL_CNT += 1

        return kpi

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=opt.seed))
    study.optimize(objective, n_trials=30)
    logger.info(f"Best trial for {opt.model}: {study.best_trial.number}")
    tune_log_path = './tune_log/sample_150/'
    f = open(tune_log_path + f'best_params_{opt.dataset}_{opt.model}.txt', 'a', encoding='utf-8')
    save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Saved at: {save_time}\n")
    f.write(f"Best trial getting the best NDCG@{opt.topK} for {opt.model}: {study.best_trial.number}\n")
    f.write(f"Best params for {opt.model}: {study.best_trial.params}\n")
    f.write(f"Best value for {opt.model}: {study.best_value}\n")
    f.flush()
    f.close()



