import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from .functions import pad_zero_for_seq, build_seqs, get_seq_from_df, build_seqs_NoID
import scipy.sparse as sp 

class NARMDataset(Dataset):
    def __init__(self, data, conf, candidate_set=None, isTrain=True):
        '''
        Session sequences dataset class

        Parameters
        ----------
        data : pd.DataFrame
            dataframe by Data.py
        logger : logging.logger
            Logger used for recording process
        '''     
        # self.data is list of [[seqs],[targets]]   
        self.data = build_seqs_NoID(data, conf['session_len'], isTrain=isTrain)
        self.candidate_set = candidate_set

    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        if self.candidate_set is not None:
            candidate_sets = self.candidate_set[index]
        else:
            candidate_sets = []
        return [torch.tensor(session_items), torch.tensor(target_item), torch.tensor(candidate_sets)]

    def __len__(self):
        return len(self.data[1])


    def get_loader(self, args, shuffle=True):
        loader = DataLoader(
            self, 
            batch_size=args['batch_size'], 
            shuffle=shuffle, 
            collate_fn=pad_zero_for_seq
        )

        return loader


class ConventionDataset(object):
    def __init__(self, data, conf):
        self.seq_data = build_seqs(get_seq_from_df(data, conf), conf['session_len'])

    def __iter__(self):
        seqs = self.seq_data[0]
        tar = self.seq_data[1]
        sess = self.seq_data[2]
        for i in range(len(tar)):
            yield seqs[i], tar[i], sess[i]
            
class GCEDataset(Dataset):
    def __init__(self, data, conf, train_len=None, candidate_set=None, isTrain=True):
        data = build_seqs_NoID(data, conf['session_len'], isTrain=isTrain)
        inputs, mask, max_len = self.handle_data(data[0], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.candidate_set = candidate_set
        
    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        if self.candidate_set is not None:
            candidate_set = self.candidate_set[index]
        else:
            candidate_set = []

        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), torch.tensor(candidate_set)]

    def __len__(self):
        return self.length
        
    def handle_data(self, inputData, train_len=None):
        len_data = [len(nowData) for nowData in inputData]
        if train_len is None:
            max_len = max(len_data)
        else:
            max_len = train_len
        # reverse the sequence
        us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
                   for upois, le in zip(inputData, len_data)]
        us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
                   for le in len_data]
        return us_pois, us_msks, max_len

class HIDEDataset(Dataset):
    def __init__(self, data, conf, train_len=None, candidate_set=None, isTrain=True):
        data = build_seqs_NoID(data, conf['session_len'], isTrain=isTrain)
        inputs, mask, max_len, max_edge_num = self.handle_data(data[0], conf['w'], conf)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.max_edge_num = max_edge_num
        self.sw = conf['w']
        self.conf = conf
        self.candidate_set = candidate_set
        
    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        if self.candidate_set is not None:
            candidate_set = self.candidate_set[index]
        else:
            candidate_set = []

        max_n_node = self.max_len
        max_n_edge = self.max_edge_num # max hyperedge num

        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        # H_s shape: (max_n_node, max_n_edge)
        rows = []
        cols = []
        vals = []
        # generate slide window hyperedge
        edge_idx = 0
        sw_set = []
        for i in range(2, self.sw):
            sw_set.append(i)
        if self.conf['sw_edge']:
            for win in sw_set:
                for i in range(len(u_input)-win+1):
                    if i+win <= len(u_input):
                        if u_input[i+win-1] == 0:
                            break
                        for j in range(i, i+win):
                            rows.append(np.where(node == u_input[j])[0][0])
                            cols.append(edge_idx)
                            vals.append(1.0)
                        edge_idx += 1
        

        if self.conf['item_edge']:
            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        if u_input[i] == item and i > 0:
                            rows.append(np.where(node == u_input[i-1])[0][0])
                            cols.append(edge_idx)
                            vals.append(2.0)
                    rows.append(np.where(node == item)[0][0])
                    cols.append(edge_idx)
                    vals.append(2.0)
                    edge_idx += 1
        
        # intent hyperedges are dynamic generated in layers.py
        u_Hs = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
        Hs = np.asarray(u_Hs.todense())
        
        return [torch.tensor(alias_inputs), torch.tensor(Hs), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), torch.tensor(candidate_set)]

    def __len__(self):
        return self.length
        
    def handle_data(self, inputData, sw, conf):
        w_set = []
        for i in range(2, sw):
            w_set.append(i)
        sw = w_set
        items, len_data = [], []
        for nowData in inputData:
            len_data.append(len(nowData))
            Is = []
            for i in nowData:
                Is.append(i)
            items.append(Is)
        # len_data = [len(nowData) for nowData in inputData]
        max_len = max(len_data)

        edge_lens = []
        for item_seq in items:
            item_num = len(list(set(item_seq)))
            num_sw = 0
            if conf['sw_edge']:
                for win_len in sw:
                    temp_num = len(item_seq) - win_len + 1
                    num_sw += temp_num
            edge_num = num_sw
            if conf['item_edge']:
                edge_num += item_num
            edge_lens.append(edge_num)

        max_edge_num = max(edge_lens)
        # reverse the sequence
        # reverse the sequence
        us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
                for upois, le in zip(inputData, len_data)]
        us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
                for le in len_data]

        #print(max_len, max_edge_num)

        return us_pois, us_msks, max_len, max_edge_num
        
class SRGNNDataset(object):
    def __init__(self, data, conf, shuffle=False, graph=None):
        data = build_seqs(get_seq_from_df(data, conf), conf['session_len'])
        inputs = data[0]
        inputs, mask, len_max = self.data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def data_masks(self, all_usr_pois, item_tail):
        us_lens = [len(upois) for upois in all_usr_pois]
        len_max = max(us_lens)
        us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
        us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
        return us_pois, us_msks, len_max

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets

class GRU4RECDataset(object):
    def __init__(self, data, conf, batch_size, time_sort=False):
        # this need item_id start from 0
        self.df = data.copy()
        self.batch_size = batch_size
        self.session_key = conf['session_key']
        self.item_key = conf['item_key']
        self.time_key = conf['time_key']
        self.time_sort = time_sort

        self.df.sort_values([self.session_key, self.time_key], inplace=True)
        self.click_offsets = self._get_click_offset()
        self.session_idx_arr = self._order_session_idx()


    def _get_click_offset(self):
        self.batch_lim = self.df[self.session_key].nunique()
        offsets = np.zeros(self.batch_lim + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def _order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    def __iter__(self):
        '''
        Returns the iterator for producing session-parallel training mini-batches.

        Yields
        -------
        input : torch.FloatTensor
            Item indices that will be encoded as one-hot vectors later. size (B,)
        target : torch.teFloatTensornsor
            a Variable that stores the target item indices, size (B,)
        masks : Numpy.array
            indicating the positions of the sessions to be terminated
        '''    
        click_offsets = self.click_offsets
        session_idx_arr = self.session_idx_arr
        iters = np.arange(self.batch_size)
        maxiter = iters.max()

        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = self.df[self.item_key].values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = self.df[self.item_key].values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

class AttMixerDataset(Dataset):
    def __init__(self, data, conf, shuffle=False, graph=None, order=2, candidate_set=None, isTrain=True):
        '''
        Session sequences dataset class

        Parameters
        ----------
        data : pd.DataFrame
            dataframe by Data.py
        logger : logging.logger
            Logger used for recording process
        '''     
        # self.data is list of [[seqs],[targets]]   
        self.data = build_seqs_NoID(data, conf['session_len'], isTrain=isTrain)
        if candidate_set is not None:
            self.candidate_set = np.asarray(candidate_set)
        else:
            self.candidate_set = candidate_set
        inputs, mask, len_max = self.data_masks(self.data[0], [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.squeeze(np.asarray(self.data[1]))
        # np.asarray(self.data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        self.order = order

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # return index
        return self.get_slice(index)
    #    return 

    


    def get_loader(self, args, shuffle=True):
        # loader = DataLoader(
        #     self, 
        #     batch_size=args['batch_size'], 
        #     shuffle=shuffle, 
        #     collate_fn=pad_zero_for_seq
        # )
        sampler = BatchSampler(SequentialSampler(self), args['batch_size'], drop_last=False)
        loader = DataLoader(self, sampler=sampler, num_workers=4, pin_memory=True)
        return loader
    
    def data_masks(self, all_usr_pois, item_tail, maxl=200):
        us_lens = [len(upois) for upois in all_usr_pois]
        len_max = max(us_lens)
        us_pois = [upois + item_tail * (len_max - le + 1) for upois, le in zip(all_usr_pois, us_lens)]
        us_msks = [[1] * le + [0] * (len_max - le + 1) for le in us_lens]
        
        return us_pois, us_msks, len_max
    
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        if self.candidate_set is not None:
            candidate_set = self.candidate_set[i]
        else:
            candidate_set = []

        items, n_node, A, masks, alias_inputs = [], [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        
        max_n_node = np.max(n_node) 
        l_seq = np.sum(mask, axis=1)
        # l_seq = np.sum(mask)
        max_l_seq = mask.shape[1]
        # max_l_seq = len(mask)
        max_n_node_aug = max_n_node
        for k in range(self.order-1):
            max_n_node_aug += max_l_seq - 1 - k
        for idx, u_input in enumerate(inputs):
            node = np.array(np.unique(u_input)[1:].tolist() + [0])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node_aug, max_n_node_aug))
            mask1 = np.zeros(max_n_node_aug)
            for i in np.arange(len(u_input)):
                if u_input[i + 1] == 0:
                    if i == 0:
                        mask1[0] = 1
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                mask1[u] = 1
                mask1[v] = 1
                u_A[u][v] += 1
                
                for t in range(self.order-1):
                    if i == 0:
                        k = max_n_node + t * max_l_seq - sum(list(range(t+1))) + i
                        mask1[k] = 1
                    if i < l_seq[idx] - t - 2:
                        k = max_n_node + t * max_l_seq - sum(list(range(t+1))) + i + 1
                        u_A[u][k] += 1
                        u_A[k-1][k] += 1
                        mask1[k] = 1
                    if i < l_seq[idx] - t - 2:
                        l = np.where(node == u_input[i + t + 2])[0][0]
                        if l is not None and l > 0:
                            u_A[k-1][l] += 1
                            mask1[l] = 1
                
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            masks.append(mask1)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        alias_inputs = torch.tensor(alias_inputs).long()
        # A = torch.tensor(A).float()
        items = torch.tensor(items).long()
        mask = torch.tensor(mask).long()
        # mask1 = torch.tensor(masks).long()
        targets = torch.tensor(targets).long()
        n_node = torch.tensor(n_node).long()

        A = torch.from_numpy(np.asarray(A)).float()
        mask1 = torch.from_numpy(np.asarray(masks)).long()
        
        candidate_set = torch.tensor(candidate_set).long()
        return alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set

