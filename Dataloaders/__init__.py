import torch

import numpy as np
from Data.preprocess import load_amazon_data
from sklearn.model_selection import train_test_split

"""target alpha"""
def split_source_target(data_name, data_path, target_idx, device):
    
    domain_names, data_insts, data_labels = load_amazon_data(data_name, data_path)
    
    # Build source instances
    if data_name == 'amazon1':
        num_trains = 2000
        Xs = [torch.tensor(data_insts[i][:num_trains, :].todense().astype(np.float32)).to(device) for i in range(len(domain_names)) if i != target_idx]
        ys = [torch.tensor(data_labels[i][:num_trains, :].ravel().astype(np.int64)).to(device) for i in range(len(domain_names)) if i != target_idx]
        Xt = torch.tensor(data_insts[target_idx][:num_trains, :].todense().astype(np.float32)).to(device)
        yt = torch.tensor(data_labels[target_idx][:num_trains, :].ravel().astype(np.int64)).to(device)
        
    elif data_name == 'amazon3':
        num_trains = 6000
        Xs = [torch.tensor(data_insts[i][:num_trains, :].astype(np.float32)).to(device) for i in range(len(domain_names)) if i != target_idx]
        ys = [torch.tensor(data_labels[i][:num_trains].astype(np.int64)).to(device) for i in range(len(domain_names)) if i != target_idx]
        Xt_train = torch.tensor(data_insts[target_idx][:num_trains, :].astype(np.float32)).to(device)
        yt_train = torch.tensor(data_labels[target_idx][:num_trains].astype(np.int64)).to(device)
        Xt_test = torch.tensor(data_insts[target_idx][num_trains:, :].astype(np.float32)).to(device)
        yt_test = torch.tensor(data_labels[target_idx][num_trains:].astype(np.int64)).to(device)
    
    else:
        NotImplementedError

    return domain_names, Xs, ys, Xt, yt


def data_loader(inputs, targets, batch_size, seed, shuffle=True):
    if seed is not None:
        np.random.seed(seed)

    assert inputs.shape[0] == targets.shape[0]
    inputs_size = inputs.shape[0]
    if shuffle:
        random_order = np.arange(inputs_size)
        np.random.shuffle(random_order)
        inputs, targets = inputs[random_order, :], targets[random_order]
    num_blocks = int(inputs_size / batch_size)
    for i in range(num_blocks):
        yield inputs[i * batch_size: (i+1) * batch_size, :], targets[i * batch_size: (i+1) * batch_size]
    if num_blocks * batch_size != inputs_size:
        yield inputs[num_blocks * batch_size:, :], targets[num_blocks * batch_size:]



def multi_data_loader(inputs, target, batch_size, seed, shuffle=True):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    input_size = [inputs[i].size(0) for i in range(len(inputs))]
    max_input_size = max(input_size)
    n_sources = len(inputs)

    if shuffle:
        for i in range(n_sources):
            r_order = np.arange(input_size[i])
            np.random.shuffle(r_order)
            inputs[i], target[i] = inputs[i][r_order], target[i][r_order]
    
    num_blocks = int(max_input_size/batch_size)
    for _ in range(num_blocks):
        xs, ys = [], []
        for i in range(n_sources):
            ridx = np.random.choice(input_size[i], batch_size)
            xs.append(inputs[i][ridx])
            ys.append(target[i][ridx])
        yield xs, ys