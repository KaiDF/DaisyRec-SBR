Dataset_setting = {
    'amazon':{
        'num_node':14240
    },
    'games':{
        'num_node':17389
    },
    'cd':{
        'num_node':65785
    },
    'ml1m':{
        'num_node':3416
    }
}

Model_setting = {
    'GCE-GNN': {
        'model_dir': 'gcegnn',
        'dataloader':'GCEDataset',
        'activate': 'relu',
        'n_sample_all': 12,
        'n_sample': 12,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'dropout_global': 0.5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'GCE-GNN',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'n_iter':1,
        'dropout_gcn':0,
        'dropout_local':0.5
    },
    'MCPRN': {
        'model_dir': 'mcprn_v4_block',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'MCPRN',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16,
        'tau':0.01,
        'purposes':2
    },
    'STAMP': {
        'model_dir': 'stamp',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'STAMP',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16,
    },
    'NARM': {
        'model_dir': 'narm',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'NARM',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16,
        'hidden_size':100,
        'n_layers':1
    },
    'FPMC': {
        'model_dir': 'conventions',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'FPMC',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16
    },
    'HIDE': {
        'model_dir': 'hide',
        'dataloader':'HIDEDataset',
        'activate': 'relu',
        'n_sample_all': 12,
        'n_sample': 12,
        'n_iter':1,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'n_layers': 1,
        'dropout_global': 0.5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'HIDE',
        'session_len': 50,
        'e':0.4,
        'disen':False, # need to be fixed
        'norm':True,
        'sw_edge': True,
        'item_edge': True,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'n_factor':3,
        'dropout_gcn':0,
        'dropout_local':0.5,
        'w':5,
        'lamda':0.01,
        'reg':1e-5
    },
    'AttenMixer': {
        'model_dir': 'attenMixer',
        'dataloader':'AttMixerDataset',
        'norm': True,
        'scale': True,
        'use_lp_pool': True,
        'softmax':True,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'n_layers': 1,
        'dropout': 0.1,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'HIDE',
        'session_len': 50,
        'dot':True,
        'last_k':7, # need to be fixed
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'l_p':3,
        'heads':8
    }

}


HyperParameter_setting = {
    'GCE-GNN': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'n_iter': [1, 2],
            'dropout_gcn': [0, 0.2, 0.4, 0.6, 0.8],
            'dropout_local': [0, 0.5],
        }
    },
    'MCPRN': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'tau': [0.01, 0.1, 1, 10],
            'purposes': [1, 2, 3, 4]
        }
    },
    'STAMP': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
        }
    },
    'NARM': {
        'int': {
            'hidden_size': {'min': 50, 'max': 200, 'step': 50}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'n_layers': [1, 2, 3]
        }
    },
    'HIDE': {
        'int': {
            'w': {'min':1, 'max': 10, 'step': 1}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'reg': [1e-5, 1e-4, 1e-3, 1e-2],
            'dropout_gcn': [0, 0.2, 0.4, 0.6, 0.8],
            'dropout_local': [0, 0.5],
            'n_factor': [1, 3, 5, 7, 9],
            'lamda': [1e-5, 1e-4, 1e-3, 1e-2]
        }
    },
    'FPMC': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
        }
    },
     'AttenMixer': {
        'int': {
            'l_p': {'min':1, 'max': 10, 'step': 1}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'heads': [1,2,4,8]
        }
    }
}

Best_setting = {
    'GCE-GNN': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
        # need to be tuned
    },
    'MCPRN': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
        # need to be tuned
   
    },
    'STAMP': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
    },
    'NARM': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
    },
    'FPMC': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
    },
    'HIDE': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
    },
    'AttenMixer': {
        'amazon':{

        },
        'games':{

        },
        'ml1m':{

        }
    }

}

