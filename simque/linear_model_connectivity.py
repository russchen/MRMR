import torch
import numpy as np
import torch.nn as nn
from data_loader import get_data_loader
import gc
from tqdm import tqdm

def train_linear_model(model_prev, model_curr, model_lmc, mem_set, train_set, config):
    print('training linear model')
    w_prev = flatten_params(model_prev, numpy_output=True)
    w_curr = flatten_params(model_curr, numpy_output=True)
    
    model_lmc = assign_weights(model_lmc, w_prev + 0.5 * (w_curr - w_prev)).to(config['device'])
    
    optimizer = torch.optim.SGD(model_lmc.parameters(), lr=config['lmc_lr'], momentum=config['momentum'])
    factor = 1
    for epoch in range(factor * config['lmc_epochs']):
        model_lmc.train()
        optimizer.zero_grad()
        grads = get_line_loss(w_prev, flatten_params(model_lmc), mem_set, config) \
                + get_line_loss(w_curr, flatten_params(model_lmc), train_set, config)
        model_lmc = assign_grads(model_lmc, grads).to(config['device'])
        optimizer.step()
    print('end training')
    return model_lmc
    
def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
        total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params
    
def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'runing_mean' in param or 'runing_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(torch.from_numpy(weights[index: index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m
    
def assign_grads(m, grads):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        if 'runing_mean' in param or 'runing_var' in param or 'num_batches_tracked' in param or state_dict[param].grad is None:
            continue
        # print(param, index)
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad = grads[index: index+param_count].view(param_shape).clone()
        index += param_count
    m.load_state_dict(state_dict)
    return m
    
def get_clf_loss(config, model, mem_set):
    data_loader = get_data_loader(config, mem_set)
    #model.eval()
    criterion = nn.CrossEntropyLoss().to(config['device'])
    test_loss = 0
    count = 0
    
    for labels, neg_labels, sentences, lengths in data_loader:
        count += len(labels)
        logits, rep = model(sentences, lengths)
        # logits_proto = model.mem_forward(rep)
        labels = labels.to(config['device'])
        loss = criterion(logits, labels)
        test_loss += loss
    test_loss /= count
    del data_loader
    gc.collect()
    return test_loss
    
def get_line_loss(start_w, w, train_set, config):
    interpolation = None
    if 'line' in config['lmc_interpolation'] or 'integral' in config['lmc_interpolation']:
        interpolation = 'linear'
    elif 'stochastic' in config['lmc_interpolation']:
        interpolation = 'stochastic'
    else:
        raise Exception('non-implemented interpolation')
        
    m = torch.load('model_lmc.pth')
    # m.set_memorized_prototypes(current_proto)
    total_loss = 0
    accum_grad = None
    criterion = nn.CrossEntropyLoss().to(config['device'])
    if interpolation == 'linear':
        for t in tqdm(np.arange(0.0, 1.01, 1.0/float(config['lmc_line_samples'])), desc="Get line loss"):
            grads = []
            cur_weight = start_w + (w - start_w) * t
            m = assign_weights(m, cur_weight).to(config['device'])
            # m.set_memorized_prototypes(current_proto)
            current_loss = get_clf_loss(config, m, train_set)
            current_loss.backward()
            for name, param in m.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad += grads
        del m
        return accum_grad
        
    elif interpolation == 'stochastic':
        data_loader = get_data_loader(config, train_set)
        for labels, neg_labels, sentences, lengths in data_loader:
            grads = []
            t = np.random.uniform()
            cur_weight = start_w + (w - start_w) * t
            m = assign_weights(m, cur_weight).to(config['device'])
            m.eavl()
            logits, _ = m(sentences, lengths)
            labels = labels.to(config['device'])
            current_loss = criterion(logits, labels)
            current_loss.backward()
            for name, param in m.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad += grads
        del m
        return accum_grad
    else:
        del m 
        return None
